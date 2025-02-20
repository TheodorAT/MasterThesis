* Min  - y
* s.t. 0 <= y <= 4
*      0 <= x 
*      0 <= y 
NAME trivial_lp_model
ROWS
 N  OBJ
 L  con
COLUMNS
     y        con      1
     y        OBJ      -1
RHS
    rhs       con      4
RANGES
BOUNDS
 LO bounds    y        0
ENDATA
