
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
    #if defined(DEAL_II_WITH_PETSC) && !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
    using namespace dealii::LinearAlgebraPETSc;
    #  define USE_PETSC_LA
    #elif defined(DEAL_II_WITH_TRILINOS)
    using namespace dealii::LinearAlgebraTrilinos;
    #else
    #  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
    #endif
}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/grid/manifold.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor_deprecated.h>

#include <deal.II/lac/sparse_ilu.h>

#include <iostream>
#include <fstream>
#include <sstream>

//hello-1


namespace Diffusion
{
    using namespace dealii;
    
    template <int dim>
    class Heat
    {        
    private:
        MPI_Comm                                  mpi_communicator;
        double deltat = 0.01;
        double totaltime = 7200;
        double conductivity = 0.01;

        int meshrefinement = 0;
        int degree;
        parallel::distributed::Triangulation<dim> triangulation;
        LA::MPI::SparseMatrix                     system_matrix;

        DoFHandler<dim>                           dof_handler;
        FESystem<dim>                             fe;
        LA::MPI::Vector                           lr_solution;
        LA::MPI::Vector                           lo_system_rhs, lo_initial_condition; 
        AffineConstraints<double>                 constraints;
        IndexSet                                  owned_partitioning;
        IndexSet                                  relevant_partitioning;
        ConditionalOStream                        pcout;
        TimerOutput                               computing_timer;
        
    public:
        void setup_system();
        void reinit_constraints();
        void assemble_system();
        void assemble_rhs();
        void solve();
        void output_results (int);
        void timeloop();
        double compute_errors();
        
        Heat(int degreein)
        :
        mpi_communicator (MPI_COMM_WORLD),
        degree(degreein),
        triangulation (mpi_communicator),
        dof_handler(triangulation),
        fe(FE_Q<dim>(degree), 1),
        pcout (std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
        computing_timer (mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)
        {      
            pcout << "constructor success...."<< std::endl;
        }
    };

    //=========================================
    template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
        RightHandSide() : Function<dim>(1) {}
        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;
    };
    
    template <int dim>
    double RightHandSide<dim>::value(const Point<dim> &, const unsigned int) const
    {
        return 0.0;
    }
    
    //==========================================
    template <int dim>
    class InitialValues : public Function<dim>
    {
    public:
        InitialValues () : Function<dim>(1) {}
        virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
    };
    
    template <int dim>
    double InitialValues<dim>::value (const Point<dim> &p, const unsigned int) const
    {
        if((pow((p[0]-0.5), 2) + pow((p[1]-0.5), 2)) < 0.2)
            return 40.0;
        else
            return 20.0;
    }
    //==========================================
    template <int dim>
    void Heat<dim>::setup_system()
    {  
        TimerOutput::Scope t(computing_timer, "setup_system");
        pcout <<"in setup_system "<<std::endl;
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(triangulation);
        std::ifstream input_file("domain_hex.msh");
        grid_in.read_msh(input_file);
        triangulation.refine_global (meshrefinement);
        dof_handler.distribute_dofs(fe);
        
        pcout << "   Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << "   Total number of cells: "
        << triangulation.n_cells()
        << std::endl;
        pcout << "   Number of degrees of freedom: "
        << dof_handler.n_dofs()
        << std::endl;
    
        pcout << "dofspercell "<< fe.dofs_per_cell << std::endl;
        
        owned_partitioning = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs (dof_handler, relevant_partitioning);
                
        {
            constraints.clear();
            constraints.reinit(relevant_partitioning);
            DoFTools::make_hanging_node_constraints(dof_handler, constraints);
            VectorTools::interpolate_boundary_values (dof_handler, 101, ConstantFunction<dim>(20), constraints);
            VectorTools::interpolate_boundary_values (dof_handler, 102, ConstantFunction<dim>(20), constraints);
            VectorTools::interpolate_boundary_values (dof_handler, 103, ConstantFunction<dim>(20), constraints);
            VectorTools::interpolate_boundary_values (dof_handler, 104, ConstantFunction<dim>(20), constraints);
            constraints.close();
        }

        system_matrix.clear();        
        DynamicSparsityPattern dsp (relevant_partitioning);
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
        SparsityTools::distribute_sparsity_pattern (dsp, dof_handler.n_locally_owned_dofs_per_processor(), mpi_communicator, relevant_partitioning);
        
        system_matrix.reinit (owned_partitioning, owned_partitioning, dsp, mpi_communicator);       
        lr_solution.reinit(owned_partitioning, relevant_partitioning, mpi_communicator);
        lo_system_rhs.reinit(owned_partitioning, mpi_communicator);
        lo_initial_condition.reinit(owned_partitioning, mpi_communicator);
        
        InitialValues<dim> initialcondition;
        VectorTools::interpolate(dof_handler, initialcondition, lo_initial_condition);
        lr_solution = lo_initial_condition;
        pcout <<"end of setup_system "<<std::endl;
    }
        //==========================================
    template <int dim>
    void Heat<dim>::reinit_constraints()
    {  
                
        {
            constraints.clear();
            constraints.reinit(relevant_partitioning);
            DoFTools::make_hanging_node_constraints(dof_handler, constraints);
            VectorTools::interpolate_boundary_values (dof_handler, 101, ConstantFunction<dim>(20), constraints);
            VectorTools::interpolate_boundary_values (dof_handler, 102, ConstantFunction<dim>(20), constraints);
            VectorTools::interpolate_boundary_values (dof_handler, 103, ConstantFunction<dim>(20), constraints);
            VectorTools::interpolate_boundary_values (dof_handler, 104, ConstantFunction<dim>(20), constraints);
            constraints.close();
        }

//         system_matrix.clear();        
//         DynamicSparsityPattern dsp (relevant_partitioning);
//         DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
//         SparsityTools::distribute_sparsity_pattern (dsp, dof_handler.n_locally_owned_dofs_per_processor(), mpi_communicator, relevant_partitioning);
    }  
    //=========================================
    template <int dim>
    void Heat<dim>::assemble_system()
    {
        TimerOutput::Scope t(computing_timer, "assembly_system");
        pcout << "in assemble_system" << std::endl;
        system_matrix=0;
        lo_system_rhs=0;
        
        QGauss<dim>  quadrature_formula(degree+2);
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                     update_values  |
                                     update_quadrature_points |
                                     update_gradients |
                                     update_JxW_values);
        
        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();
        
        FullMatrix<double>                   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>                       local_rhs(dofs_per_cell);        
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::vector<double>                  old_values(n_q_points);
        std::vector<double>                  value_phi(dofs_per_cell);
        std::vector<Tensor<1,dim>>           gradient_phi(dofs_per_cell);

        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
        
        for (; cell!=endc; ++cell)
        {
            if (cell->is_locally_owned())
            {
                fe_values.reinit(cell);
                local_matrix = 0;
                local_rhs = 0;
                fe_values.get_function_values(lr_solution, old_values);
                
                for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
                {
                    for (unsigned int k=0; k<dofs_per_cell; ++k)
                    {
                        value_phi[k]   = fe_values.shape_value (k, q_index);
                        gradient_phi[k]= fe_values.shape_grad (k, q_index);
                    }
                    for (unsigned int i=0; i<dofs_per_cell; ++i)            
                    {
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        {                            
                            local_matrix(i,j) += (value_phi[i]*value_phi[j] + deltat*conductivity* gradient_phi[i]*gradient_phi[j])*fe_values.JxW(q_index);
                        }
                           local_rhs(i) += (old_values[q_index]*value_phi[i])*fe_values.JxW(q_index);
                    }
                } //end of quadrature points loop
                cell->get_dof_indices(local_dof_indices);
                constraints.distribute_local_to_global(local_matrix, local_rhs, local_dof_indices, system_matrix, lo_system_rhs);
//                 constraints.distribute_local_to_global(local_matrix, local_dof_indices, system_matrix);
//                 constraints.distribute_local_to_global(local_rhs, local_dof_indices, lo_system_rhs);
            } // end of if vof_cell->is_locally_owned()
        } //  end of cell loop
        system_matrix.compress (VectorOperation::add);
        lo_system_rhs.compress (VectorOperation::add);
    }
        //=========================================
    template <int dim>
    void Heat<dim>::assemble_rhs()
    {
        TimerOutput::Scope t(computing_timer, "assembly_rhs");
        pcout << "in assemble_rhs" << std::endl;
        lo_system_rhs=0;
        
        QGauss<dim>  quadrature_formula(degree+2);
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                     update_values  |
                                     update_quadrature_points |
                                     update_JxW_values);
        
        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();

        Vector<double>                       local_rhs(dofs_per_cell);        
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::vector<double>                  old_values(n_q_points);
        std::vector<double>                  value_phi(dofs_per_cell);
        
        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
        
        for (; cell!=endc; ++cell)
        {
            if (cell->is_locally_owned())
            {
                fe_values.reinit(cell);
                local_rhs = 0;
                fe_values.get_function_values(lr_solution, old_values);
                
                for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
                {                    
                    for (unsigned int i=0; i<dofs_per_cell; ++i)            
                    {
                           local_rhs(i) += (old_values[q_index]*fe_values.shape_value(i, q_index))*fe_values.JxW(q_index);
                    }
                } //end of quadrature points loop
                cell->get_dof_indices(local_dof_indices);
                constraints.distribute_local_to_global(local_rhs, local_dof_indices, lo_system_rhs);
            } // end of if vof_cell->is_locally_owned()
        } //  end of cell loop
        lo_system_rhs.compress (VectorOperation::add);
    }
    //=========================================
    template <int dim>
    void Heat<dim>::solve()
    {
        pcout <<"in solve"<<std::endl;
        TimerOutput::Scope t(computing_timer, "solve");
        LA::MPI::Vector  distributed_solution (owned_partitioning, mpi_communicator);
        
        SolverControl solver_control(dof_handler.n_dofs(), 1e-12);
        dealii::PETScWrappers::SparseDirectMUMPS solver(solver_control, mpi_communicator);
        
        solver.solve (system_matrix, distributed_solution, lo_system_rhs);
        constraints.distribute(distributed_solution);
        lr_solution = distributed_solution;

        pcout <<"end of solve"<<std::endl;
    }
        //=========================================
        template <int dim>
        double Heat<dim>::compute_errors()
        {      
            Vector<double> cellwise_errors(triangulation.n_active_cells());
            QGauss<dim> quadrature(4);
            VectorTools::integrate_difference (dof_handler, lr_solution, ZeroFunction<dim>(1), cellwise_errors, quadrature, VectorTools::L2_norm);
            const double u_l2_error = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L2_norm);
            return u_l2_error;
        }
    //=========================================
    template <int dim>
    void Heat<dim>::output_results(int timestepnumber)
    {
        TimerOutput::Scope t(computing_timer, "output");
        
        DataOut<dim> data_out;
        data_out.add_data_vector(dof_handler, lr_solution, "temp");
        Vector<float> subdomain (triangulation.n_active_cells());
        for (unsigned int i=0; i<subdomain.size(); ++i)
            subdomain(i) = triangulation.locally_owned_subdomain();
        data_out.add_data_vector (subdomain, "subdomain");
        data_out.build_patches ();
        
        const std::string filename = ("zzso" + Utilities::int_to_string (timestepnumber, 3) + "." +Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
        std::ofstream output ((filename + ".vtu").c_str());
        data_out.write_vtu (output);
        
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            std::vector<std::string> filenames;
            for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
                filenames.push_back ("zzso" + Utilities::int_to_string (timestepnumber, 3) + "." + Utilities::int_to_string (i, 4) + ".vtu");
            
            std::ofstream master_output (("zzso" + Utilities::int_to_string (timestepnumber, 3) + ".pvtu").c_str());
            data_out.write_pvtu_record (master_output, filenames);
        }
    }
    //=========================================  
    template <int dim>
    void Heat<dim>::timeloop()
    {      
        double timet = 0;
        int timestepnumber=0;
//         assemble_system();
//         assemble_rhs();
//         output_results(timestepnumber);
        while(timet<totaltime)
        {
            output_results(timestepnumber);
            timet+=deltat;
            timestepnumber++;
            pcout <<"timet "<<timet <<std::endl;
            assemble_system();
//             assemble_rhs();
            pcout << "l2_error " << compute_errors() << std::endl;
            solve();
        }
    }
}  // end of namespace
//=============================================

int main (int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace Diffusion;  
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);        
        Heat<2> problem(1);   
        problem.setup_system();
        problem.timeloop();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }    
    return 0;
}
