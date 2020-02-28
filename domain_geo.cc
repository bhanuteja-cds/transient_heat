lc1 = 0.1;
xmax = 1;
ymax = 1;

Point(1) = {0,0,0,lc1};
Point(2) = {xmax,0,0,lc1};
Point(3) = {xmax,ymax,0,lc1};
Point(4) = {0,ymax,0,lc1};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
 
Line Loop(1) = {1:4};
Plane Surface(1) = {1};

Physical Line(101) = {1};
Physical Line(102) = {2};
Physical Line(103) = {3};
Physical Line(104) = {4};
Physical Surface(150)={1};

