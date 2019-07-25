// Gmsh project created on Thu Jul 25 21:41:25 2019
SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 1, 1, 0};
Disk(2) = {0.5, 0.5, 0, 0.25, 0.25};
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; }
Transfinite Line {9, 8, 6, 7} = 30 Using Progression 1;
Mesh.CharacteristicLengthMin=0.3;
Mesh.CharacteristicLengthMax=0.3;
