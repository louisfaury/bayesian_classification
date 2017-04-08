 • Implementation of basic bayesian (and GML to be compared with) methods for binary classification on the *Breast Cancer (Diagnostic) Wiscounsin* dataset.  
 • The full dataset is composed of 699 points laying in a 10-dimensional space. It is detailed below : 


    Radius    Texture    Perimeter    Area    Smoothness    Compactness    Concacity    Concavity    Symetry    Diagnostic
    ______    _______    _________    ____    __________    ___________    _________    _________    _______    __________

     5         1          1            1       2             1              3            1            1         'benign'  
     5         4          4            5       7            10              3            2            1         'benign'  
     3         1          1            1       2             2              3            1            1         'benign'  
     6         8          8            1       3             4              3            7            1         'benign'  
     4         1          1            3       2             1              3            1            1         'benign'  
     8        10         10            8       7            10              9            7            1         'malign'  
     1         1          1            1       2            10              3            1            1         'benign'  
     2         1          2            1       2             1              3            1            1         'benign'  
     2         1          1            1       2             1              1            1            5         'benign'  
     4         2          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       1             1              3            1            1         'benign'  
     2         1          1            1       2             1              2            1            1         'benign'  
     5         3          3            3       2             3              4            4            1         'malign'  
     1         1          1            1       2             3              3            1            1         'benign'  
     8         7          5           10       7             9              5            5            4         'malign'  
     7         4          6            4       6             1              4            3            1         'malign'  
     4         1          1            1       2             1              2            1            1         'benign'  
     4         1          1            1       2             1              3            1            1         'benign'  
    10         7          7            6       4            10              4            1            2         'malign'  
     6         1          1            1       2             1              3            1            1         'benign'  
     7         3          2           10       5            10              5            4            4         'malign'  
    10         5          5            3       6             7              7           10            1         'malign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     8         4          5            1       2             0              7            3            1         'malign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     5         2          3            4       2             7              3            6            1         'malign'  
     3         2          1            1       1             1              2            1            1         'benign'  
     5         1          1            1       2             1              2            1            1         'benign'  
     2         1          1            1       2             1              2            1            1         'benign'  
     1         1          3            1       2             1              1            1            1         'benign'  
     3         1          1            1       1             1              2            1            1         'benign'  
     2         1          1            1       2             1              3            1            1         'benign'  
    10         7          7            3       8             5              7            4            3         'malign'  
     2         1          1            2       2             1              3            1            1         'benign'  
     3         1          2            1       2             1              2            1            1         'benign'  
     2         1          1            1       2             1              2            1            1         'benign'  
    10        10         10            8       6             1              8            9            1         'malign'  
     6         2          1            1       1             1              7            1            1         'benign'  
     5         4          4            9       2            10              5            6            1         'malign'  
     2         5          3            3       6             7              7            5            1         'malign'  
     6         6          6            9       6             0              7            8            1         'benign'  
    10         4          3            1       3             3              6            5            2         'malign'  
     6        10         10            2       8            10              7            3            3         'malign'  
     5         6          5            6      10             1              3            1            1         'malign'  
    10        10         10            4       8             1              8           10            1         'malign'  
     1         1          1            1       2             1              2            1            2         'benign'  
     3         7          7            4       4             9              4            8            1         'malign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     4         1          1            3       2             1              3            1            1         'benign'  
     7         8          7            2       4             8              3            8            2         'malign'  
     9         5          8            1       2             3              2            1            5         'malign'  
     5         3          3            4       2             4              3            4            1         'malign'  
    10         3          6            2       3             5              4           10            2         'malign'  
     5         5          5            8      10             8              7            3            7         'malign'  
    10         5          5            6       8             8              7            1            1         'malign'  
    10         6          6            3       4             5              3            6            1         'malign'  
     8        10         10            1       3             6              3            9            1         'malign'  
     8         2          4            1       5             1              5            4            4         'malign'  
     5         2          3            1       6            10              5            1            1         'malign'  
     9         5          5            2       2             2              5            1            1         'malign'  
     5         3          5            5       3             3              4           10            1         'malign'  
     1         1          1            1       2             2              2            1            1         'benign'  
     9        10         10            1      10             8              3            3            1         'malign'  
     6         3          4            1       5             2              3            9            1         'malign'  
     1         1          1            1       2             1              2            1            1         'benign'  
    10         4          2            1       3             2              4            3           10         'malign'  
     4         1          1            1       2             1              3            1            1         'benign'  
     5         3          4            1       8            10              4            9            1         'malign'  
     8         3          8            3       4             9              8            9            8         'malign'  
     1         1          1            1       2             1              3            2            1         'benign'  
     5         1          3            1       2             1              2            1            1         'benign'  
     6        10          2            8      10             2              7            8           10         'malign'  
     1         3          3            2       2             1              7            2            1         'benign'  
     9         4          5           10       6            10              4            8            1         'malign'  
    10         6          4            1       3             4              3            2            3         'malign'  
     1         1          2            1       2             2              4            2            1         'benign'  
     1         1          4            1       2             1              2            1            1         'benign'  
     5         3          1            2       2             1              2            1            1         'benign'  
     3         1          1            1       2             3              3            1            1         'benign'  
     2         1          1            1       3             1              2            1            1         'benign'  
     2         2          2            1       1             1              7            1            1         'benign'  
     4         1          1            2       2             1              2            1            1         'benign'  
     5         2          1            1       2             1              3            1            1         'benign'  
     3         1          1            1       2             2              7            1            1         'benign'  
     3         5          7            8       8             9              7           10            7         'malign'  
     5        10          6            1      10             4              4           10           10         'malign'  
     3         3          6            4       5             8              4            4            1         'malign'  
     3         6          6            6       5            10              6            8            3         'malign'  
     4         1          1            1       2             1              3            1            1         'benign'  
     2         1          1            2       3             1              2            1            1         'benign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     3         1          1            2       2             1              1            1            1         'benign'  
     4         1          1            1       2             1              3            1            1         'benign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     2         1          1            1       2             1              3            1            1         'benign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     2         1          1            2       2             1              1            1            1         'benign'  
     5         1          1            1       2             1              3            1            1         'benign'  
     9         6          9            2      10             6              2            9           10         'malign'  
     7         5          6           10       5            10              7            9            4         'malign'  
    10         3          5            1      10             5              3           10            2         'malign'  
     2         3          4            4       2             5              2            5            1         'malign'  
     4         1          2            1       2             1              3            1            1         'benign'  
     8         2          3            1       6             3              7            1            1         'malign'  
    10        10         10           10      10             1              8            8            8         'malign'  
     7         3          4            4       3             3              3            2            7         'malign'  
    10        10         10            8       2            10              4            1            1         'malign'  
     1         6          8           10       8            10              5            7            1         'malign'  
     1         1          1            1       2             1              2            3            1         'benign'  
     6         5          4            4       3             9              7            8            3         'malign'  
     1         3          1            2       2             2              5            3            2         'benign'  
     8         6          4            3       5             9              3            1            1         'malign'  
    10         3          3           10       2            10              7            3            3         'malign'  
    10        10         10            3      10             8              8            1            1         'malign'  
     3         3          2            1       2             3              3            1            1         'benign'  
     1         1          1            1       2             5              1            1            1         'benign'  
     8         3          3            1       2             2              3            2            1         'benign'  
     4         5          5           10       4            10              7            5            8         'malign'  
     1         1          1            1       4             3              1            1            1         'benign'  
     3         2          1            1       2             2              3            1            1         'benign'  
     1         1          2            2       2             1              3            1            1         'benign'  
     4         2          1            1       2             2              3            1            1         'benign'  
    10        10         10            2      10            10              5            3            3         'malign'  
     5         3          5            1       8            10              5            3            1         'malign'  
     5         4          6            7       9             7              8           10            1         'malign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     7         5          3            7       4            10              7            5            5         'malign'  
     3         1          1            1       2             1              3            1            1         'benign'  
     8         3          5            4       5            10              1            6            2         'malign'  
     1         1          1            1      10             1              1            1            1         'benign'  
     5         1          3            1       2             1              2            1            1         'benign'  
     2         1          1            1       2             1              3            1            1         'benign'  
     5        10          8           10       8            10              3            6            3         'malign'  
     3         1          1            1       2             1              2            2            1         'benign'  
     3         1          1            1       3             1              2            1            1         'benign'  
     5         1          1            1       2             2              3            3            1         'benign'  
     4         1          1            1       2             1              2            1            1         'benign'  
     3         1          1            1       2             1              1            1            1         'benign'  
     4         1          2            1       2             1              2            1            1         'benign'  
     1         1          1            1       1             0              2            1            1         'benign'  
     3         1          1            1       2             1              1            1            1         'benign'  
     2         1          1            1       2             1              1            1            1         'benign'  
     9         5          5            4       4             5              4            3            3         'malign'  
     1         1          1            1       2             5              1            1            1         'benign'  
     2         1          1            1       2             1              2            1            1         'benign'  
     1         1          3            1       2             0              2            1            1         'benign'  
     3         4          5            2       6             8              4            1            1         'malign'  
     1         1          1            1       3             2              2            1            1         'benign'  
     3         1          1            3       8             1              5            8            1         'benign'  
     8         8          7            4      10            10              7            8            7         'malign'  
     1         1          1            1       1             1              3            1            1         'benign'  
     7         2          4            1       6            10              5            4            3         'malign'  
    10        10          8            6       4             5              8           10            1         'malign'  
     4         1          1            1       2             3              1            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     5         5          5            6       3            10              3            1            1         'malign'  
     1         2          2            1       2             1              2            1            1         'benign'  
     2         1          1            1       2             1              3            1            1         'benign'  
     1         1          2            1       3             0              1            1            1         'benign'  
     9         9         10            3       6            10              7           10            6         'malign'  
    10         7          7            4       5            10              5            7            2         'malign'  
     4         1          1            1       2             1              3            2            1         'benign'  
     3         1          1            1       2             1              3            1            1         'benign'  
     1         1          1            2       1             3              1            1            7         'benign'  
     5         1          1            1       2             0              3            1            1         'benign'  
     4         1          1            1       2             2              3            2            1         'benign'  
     5         6          7            8       8            10              3           10            3         'malign'  
    10         8         10           10       6             1              3            1           10         'malign'  
     3         1          1            1       2             1              3            1            1         'benign'  
     1         1          1            2       1             1              1            1            1         'benign'  
     3         1          1            1       2             1              1            1            1         'benign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     6        10         10           10       8            10             10           10            7         'malign'  
     8         6          5            4       3            10              6            1            1         'malign'  
     5         8          7            7      10            10              5            7            1         'malign'  
     2         1          1            1       2             1              3            1            1         'benign'  
     5        10         10            3       8             1              5           10            3         'malign'  
     4         1          1            1       2             1              3            1            1         'benign'  
     5         3          3            3       6            10              3            1            1         'malign'  
     1         1          1            1       1             1              3            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     6         1          1            1       2             1              3            1            1         'benign'  
     5         8          8            8       5            10              7            8            1         'malign'  
     8         7          6            4       4            10              5            1            1         'malign'  
     2         1          1            1       1             1              3            1            1         'benign'  
     1         5          8            6       5             8              7           10            1         'malign'  
    10         5          6           10       6            10              7            7           10         'malign'  
     5         8          4           10       5             8              9           10            1         'malign'  
     1         2          3            1       2             1              3            1            1         'benign'  
    10        10         10            8       6             8              7           10            1         'malign'  
     7         5         10           10      10            10              4           10            3         'malign'  
     5         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     3         1          1            1       2             1              3            1            1         'benign'  
     4         1          1            1       2             1              3            1            1         'benign'  
     8         4          4            5       4             7              7            8            2         'benign'  
     5         1          1            4       2             1              3            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     9         7          7            5       5            10              7            8            3         'malign'  
    10         8          8            4      10            10              8            1            1         'malign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     5         1          1            1       2             1              3            1            1         'benign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     5        10         10            9       6            10              7           10            5         'malign'  
    10        10          9            3       7             5              3            5            1         'malign'  
     1         1          1            1       1             1              3            1            1         'benign'  
     1         1          1            1       1             1              3            1            1         'benign'  
     5         1          1            1       1             1              3            1            1         'benign'  
     8        10         10           10       5            10              8           10            6         'malign'  
     8        10          8            8       4             8              7            7            1         'malign'  
     1         1          1            1       2             1              3            1            1         'benign'  
    10        10         10           10       7            10              7           10            4         'malign'  
    10        10         10           10       3            10             10            6            1         'malign'  
     8         7          8            7       5             5              5           10            2         'malign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     6        10          7            7       6             4              8           10            2         'malign'  
     6         1          3            1       2             1              3            1            1         'benign'  
     1         1          1            2       2             1              3            1            1         'benign'  
    10         6          4            3      10            10              9           10            1         'malign'  
     4         1          1            3       1             5              2            1            1         'malign'  
     7         5          6            3       3             8              7            4            1         'malign'  
    10         5          5            6       3            10              7            9            2         'malign'  
     1         1          1            1       2             1              2            1            1         'benign'  
    10         5          7            4       4            10              8            9            1         'malign'  
     8         9          9            5       3             5              7            7            1         'malign'  
     1         1          1            1       1             1              3            1            1         'benign'  
    10        10         10            3      10            10              9           10            1         'malign'  
     7         4          7            4       3             7              7            6            1         'malign'  
     6         8          7            5       6             8              8            9            2         'malign'  
     8         4          6            3       3             1              4            3            1         'benign'  
    10         4          5            5       5            10              4            1            1         'malign'  
     3         3          2            1       3             1              3            6            1         'benign'  
     3         1          4            1       2             0              3            1            1         'benign'  
    10         8          8            2       8            10              4            8           10         'malign'  
     9         8          8            5       6             2              4           10            4         'malign'  
     8        10         10            8       6             9              3           10           10         'malign'  
    10         4          3            2       3            10              5            3            2         'malign'  
     5         1          3            3       2             2              2            3            1         'benign'  
     3         1          1            3       1             1              3            1            1         'benign'  
     2         1          1            1       2             1              3            1            1         'benign'  
     1         1          1            1       2             5              5            1            1         'benign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     5         1          1            2       2             2              3            1            1         'benign'  
     8        10         10            8       5            10              7            8            1         'malign'  
     8         4          4            1       2             9              3            3            1         'malign'  
     4         1          1            1       2             1              3            6            1         'benign'  
     3         1          1            1       2             0              3            1            1         'benign'  
     1         2          2            1       2             1              1            1            1         'benign'  
    10         4          4           10       2            10              5            3            3         'malign'  
     6         3          3            5       3            10              3            5            3         'benign'  
     6        10         10            2       8            10              7            3            3         'malign'  
     9        10         10            1      10             8              3            3            1         'malign'  
     5         6          6            2       4            10              3            6            1         'malign'  
     3         1          1            1       2             1              1            1            1         'benign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     3         1          1            1       2             1              3            1            1         'benign'  
     5         7          7            1       5             8              3            4            1         'benign'  
    10         5          8           10       3            10              5            1            3         'malign'  
     5        10         10            6      10            10             10            6            5         'malign'  
     8         8          9            4       5            10              7            8            1         'malign'  
    10         4          4           10       6            10              5            5            1         'malign'  
     7         9          4           10      10             3              5            3            3         'malign'  
     5         1          4            1       2             1              3            2            1         'benign'  
    10        10          6            3       3            10              4            3            2         'malign'  
     3         3          5            2       3            10              7            1            1         'malign'  
    10         8          8            2       3             4              8            7            8         'malign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     8         4          7            1       3            10              3            9            2         'malign'  
     5         1          1            1       2             1              3            1            1         'benign'  
     3         3          5            2       3            10              7            1            1         'malign'  
     7         2          4            1       3             4              3            3            1         'malign'  
     3         1          1            1       2             1              3            2            1         'benign'  
     3         1          3            1       2             0              2            1            1         'benign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              3            1            1         'benign'  
    10         5          7            3       3             7              3            3            8         'malign'  
     3         1          1            1       2             1              3            1            1         'benign'  
     2         1          1            2       2             1              3            1            1         'benign'  
     1         4          3           10       4            10              5            6            1         'malign'  
    10         4          6            1       2            10              5            3            1         'malign'  
     7         4          5           10       2            10              3            8            2         'malign'  
     8        10         10           10       8            10             10            7            3         'malign'  
    10        10         10           10      10            10              4           10           10         'malign'  
     3         1          1            1       3             1              2            1            1         'benign'  
     6         1          3            1       4             5              5           10            1         'malign'  
     5         6          6            8       6            10              4           10            4         'malign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     8         8          8            1       2             0              6           10            1         'malign'  
    10         4          4            6       2            10              2            3            1         'malign'  
     1         1          1            1       2             0              2            1            1         'benign'  
     5         5          7            8       6            10              7            4            1         'malign'  
     5         3          4            3       4             5              4            7            1         'benign'  
     5         4          3            1       2             0              2            3            1         'benign'  
     8         2          1            1       5             1              1            1            1         'benign'  
     9         1          2            6       4            10              7            7            2         'malign'  
     8         4         10            5       4             4              7           10            1         'malign'  
     1         1          1            1       2             1              3            1            1         'benign'  
    10        10         10            7       9            10              7           10           10         'malign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     8         3          4            9       3            10              3            3            1         'malign'  
    10         8          4            4       4            10              3           10            4         'malign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     7         8          7            6       4             3              8            8            4         'malign'  
     3         1          1            1       2             5              5            1            1         'benign'  
     2         1          1            1       3             1              2            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     8         6          4           10      10             1              3            5            1         'malign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     1         1          1            1       1             1              2            1            1         'benign'  
     4         6          5            6       7             0              4            9            1         'benign'  
     5         5          5            2       5            10              4            3            1         'malign'  
     6         8          7            8       6             8              8            9            1         'malign'  
     1         1          1            1       5             1              3            1            1         'benign'  
     4         4          4            4       6             5              7            3            1         'benign'  
     7         6          3            2       5            10              7            4            6         'malign'  
     3         1          1            1       2             0              3            1            1         'benign'  
     3         1          1            1       2             1              3            1            1         'benign'  
     5         4          6           10       2            10              4            1            1         'malign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     3         2          2            1       2             1              2            3            1         'benign'  
    10         1          1            1       2            10              5            4            1         'malign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     8        10          3            2       6             4              3           10            1         'malign'  
    10         4          6            4       5            10              7            1            1         'malign'  
    10         4          7            2       2             8              6            1            1         'malign'  
     5         1          1            1       2             1              3            1            2         'benign'  
     5         2          2            2       2             1              2            2            1         'benign'  
     5         4          6            6       4            10              4            3            1         'malign'  
     8         6          7            3       3            10              3            4            2         'malign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     6         5          5            8       4            10              3            4            1         'malign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     1         1          1            1       1             1              2            1            1         'benign'  
     8         5          5            5       2            10              4            3            1         'malign'  
    10         3          3            1       2            10              7            6            1         'malign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     2         1          1            1       2             1              1            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     7         6          4            8      10            10              9            5            3         'malign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     5         2          2            2       3             1              1            3            1         'benign'  
     1         1          1            1       1             1              1            3            1         'benign'  
     3         4          4           10       5             1              3            3            1         'malign'  
     4         2          3            5       3             8              7            6            1         'malign'  
     5         1          1            3       2             1              1            1            1         'benign'  
     2         1          1            1       2             1              3            1            1         'benign'  
     3         4          5            3       7             3              4            6            1         'benign'  
     2         7         10           10       7            10              4            9            4         'malign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     4         1          1            1       3             1              2            2            1         'benign'  
     5         3          3            1       3             3              3            3            3         'malign'  
     8        10         10            7      10            10              7            3            8         'malign'  
     8        10          5            3       8             4              4           10            3         'malign'  
    10         3          5            4       3             7              3            5            3         'malign'  
     6        10         10           10      10            10              8           10           10         'malign'  
     3        10          3           10       6            10              5            1            4         'malign'  
     3         2          2            1       4             3              2            1            1         'benign'  
     4         4          4            2       2             3              2            1            1         'benign'  
     2         1          1            1       2             1              3            1            1         'benign'  
     2         1          1            1       2             1              2            1            1         'benign'  
     6        10         10           10       8            10              7           10            7         'malign'  
     5         8          8           10       5            10              8           10            3         'malign'  
     1         1          3            1       2             1              1            1            1         'benign'  
     1         1          3            1       1             1              2            1            1         'benign'  
     4         3          2            1       3             1              2            1            1         'benign'  
     1         1          3            1       2             1              1            1            1         'benign'  
     4         1          2            1       2             1              2            1            1         'benign'  
     5         1          1            2       2             1              2            1            1         'benign'  
     3         1          2            1       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       1             1              2            1            1         'benign'  
     3         1          1            4       3             1              2            2            1         'benign'  
     5         3          4            1       4             1              3            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
    10         6          3            6       4            10              7            8            4         'malign'  
     3         2          2            2       2             1              3            2            1         'benign'  
     2         1          1            1       2             1              1            1            1         'benign'  
     2         1          1            1       2             1              1            1            1         'benign'  
     3         3          2            2       3             1              1            2            3         'benign'  
     7         6          6            3       2            10              7            1            1         'malign'  
     5         3          3            2       3             1              3            1            1         'benign'  
     2         1          1            1       2             1              2            2            1         'benign'  
     5         1          1            1       3             2              2            2            1         'benign'  
     1         1          1            2       2             1              2            1            1         'benign'  
    10         8          7            4       3            10              7            9            1         'malign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       1             1              1            1            1         'benign'  
     1         2          3            1       2             1              2            1            1         'benign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     3         1          1            1       2             1              3            1            1         'benign'  
     4         1          1            1       2             1              1            1            1         'benign'  
     3         2          1            1       2             1              2            2            1         'benign'  
     1         2          3            1       2             1              1            1            1         'benign'  
     3        10          8            7       6             9              9            3            8         'malign'  
     3         1          1            1       2             1              1            1            1         'benign'  
     5         3          3            1       2             1              2            1            1         'benign'  
     3         1          1            1       2             4              1            1            1         'benign'  
     1         2          1            3       2             1              1            2            1         'benign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     4         2          2            1       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     2         3          2            2       2             2              3            1            1         'benign'  
     3         1          2            1       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       1             0              2            1            1         'benign'  
    10        10         10            6       8             4              8            5            1         'malign'  
     5         1          2            1       2             1              3            1            1         'benign'  
     8         5          6            2       3            10              6            6            1         'malign'  
     3         3          2            6       3             3              3            5            1         'benign'  
     8         7          8            5      10            10              7            2            1         'malign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     5         2          2            2       2             2              3            2            2         'benign'  
     2         3          1            1       5             1              1            1            1         'benign'  
     3         2          2            3       2             3              3            1            1         'benign'  
    10        10         10            7      10            10              8            2            1         'malign'  
     4         3          3            1       2             1              3            3            1         'benign'  
     5         1          3            1       2             1              2            1            1         'benign'  
     3         1          1            1       2             1              1            1            1         'benign'  
     9        10         10           10      10            10             10           10            1         'malign'  
     5         3          6            1       2             1              1            1            1         'benign'  
     8         7          8            2       4             2              5           10            1         'malign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     2         1          1            1       2             1              2            1            1         'benign'  
     1         3          1            1       2             1              2            2            1         'benign'  
     5         1          1            3       4             1              3            2            1         'benign'  
     5         1          1            1       2             1              2            2            1         'benign'  
     3         2          2            3       2             1              1            1            1         'benign'  
     6         9          7            5       5             8              4            2            1         'benign'  
    10         8         10            1       3            10              5            1            1         'malign'  
    10        10         10            1       6             1              2            8            1         'malign'  
     4         1          1            1       2             1              1            1            1         'benign'  
     4         1          3            3       2             1              1            1            1         'benign'  
     5         1          1            1       2             1              1            1            1         'benign'  
    10         4          3           10       4            10             10            1            1         'malign'  
     5         2          2            4       2             4              1            1            1         'benign'  
     1         1          1            3       2             3              1            1            1         'benign'  
     1         1          1            1       2             2              1            1            1         'benign'  
     5         1          1            6       3             1              2            1            1         'benign'  
     2         1          1            1       2             1              1            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     5         1          1            1       2             1              1            1            1         'benign'  
     1         1          1            1       1             1              1            1            1         'benign'  
     5         7          9            8       6            10              8           10            1         'malign'  
     4         1          1            3       1             1              2            1            1         'benign'  
     5         1          1            1       2             1              1            1            1         'benign'  
     3         1          1            3       2             1              1            1            1         'benign'  
     4         5          5            8       6            10             10            7            1         'malign'  
     2         3          1            1       3             1              1            1            1         'benign'  
    10         2          2            1       2             6              1            1            2         'malign'  
    10         6          5            8       5            10              8            6            1         'malign'  
     8         8          9            6       6             3             10           10            1         'malign'  
     5         1          2            1       2             1              1            1            1         'benign'  
     5         1          3            1       2             1              1            1            1         'benign'  
     5         1          1            3       2             1              1            1            1         'benign'  
     3         1          1            1       2             5              1            1            1         'benign'  
     6         1          1            3       2             1              1            1            1         'benign'  
     4         1          1            1       2             1              1            2            1         'benign'  
     4         1          1            1       2             1              1            1            1         'benign'  
    10         9          8            7       6             4              7           10            3         'malign'  
    10         6          6            2       4            10              9            7            1         'malign'  
     6         6          6            5       4            10              7            6            2         'malign'  
     4         1          1            1       2             1              1            1            1         'benign'  
     1         1          2            1       2             1              2            1            1         'benign'  
     3         1          1            1       1             1              2            1            1         'benign'  
     6         1          1            3       2             1              1            1            1         'benign'  
     6         1          1            1       1             1              1            1            1         'benign'  
     4         1          1            1       2             1              1            1            1         'benign'  
     5         1          1            1       2             1              1            1            1         'benign'  
     3         1          1            1       2             1              1            1            1         'benign'  
     4         1          2            1       2             1              1            1            1         'benign'  
     4         1          1            1       2             1              1            1            1         'benign'  
     5         2          1            1       2             1              1            1            1         'benign'  
     4         8          7           10       4            10              7            5            1         'malign'  
     5         1          1            1       1             1              1            1            1         'benign'  
     5         3          2            4       2             1              1            1            1         'benign'  
     9        10         10           10      10             5             10           10           10         'malign'  
     8         7          8            5       5            10              9           10            1         'malign'  
     5         1          2            1       2             1              1            1            1         'benign'  
     1         1          1            3       1             3              1            1            1         'benign'  
     3         1          1            1       1             1              2            1            1         'benign'  
    10        10         10           10       6            10              8            1            5         'malign'  
     3         6          4           10       3             3              3            4            1         'malign'  
     6         3          2            1       3             4              4            1            1         'malign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     5         8          9            4       3            10              7            1            1         'malign'  
     4         1          1            1       1             1              2            1            1         'benign'  
     5        10         10           10       6            10              6            5            2         'malign'  
     5         1          2           10       4             5              2            1            1         'benign'  
     3         1          1            1       1             1              2            1            1         'benign'  
     1         1          1            1       1             1              1            1            1         'benign'  
     4         2          1            1       2             1              1            1            1         'benign'  
     4         1          1            1       2             1              2            1            1         'benign'  
     4         1          1            1       2             1              2            1            1         'benign'  
     6         1          1            1       2             1              3            1            1         'benign'  
     4         1          1            1       2             1              2            1            1         'benign'  
     4         1          1            2       2             1              2            1            1         'benign'  
     4         1          1            1       2             1              3            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     3         3          1            1       2             1              1            1            1         'benign'  
     8        10         10           10       7             5              4            8            7         'malign'  
     1         1          1            1       2             4              1            1            1         'benign'  
     5         1          1            1       2             1              1            1            1         'benign'  
     2         1          1            1       2             1              1            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     5         1          1            1       2             1              2            1            1         'benign'  
     5         1          1            1       2             1              1            1            1         'benign'  
     3         1          1            1       1             1              2            1            1         'benign'  
     6         6          7           10       3            10              8           10            2         'malign'  
     4        10          4            7       3            10              9           10            1         'malign'  
     1         1          1            1       1             1              1            1            1         'benign'  
     1         1          1            1       1             1              2            1            1         'benign'  
     3         1          2            2       2             1              1            1            1         'benign'  
     4         7          8            3       4            10              9            1            1         'malign'  
     1         1          1            1       3             1              1            1            1         'benign'  
     4         1          1            1       3             1              1            1            1         'benign'  
    10         4          5            4       3             5              7            3            1         'malign'  
     7         5          6           10       4            10              5            3            1         'malign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     3         1          1            2       2             1              1            1            1         'benign'  
     4         1          1            1       2             1              1            1            1         'benign'  
     4         1          1            1       2             1              3            1            1         'benign'  
     6         1          3            2       2             1              1            1            1         'benign'  
     4         1          1            1       1             1              2            1            1         'benign'  
     7         4          4            3       4            10              6            9            1         'malign'  
     4         2          2            1       2             1              2            1            1         'benign'  
     1         1          1            1       1             1              3            1            1         'benign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     2         1          1            1       2             1              2            1            1         'benign'  
     1         1          3            2       2             1              3            1            1         'benign'  
     5         1          1            1       2             1              3            1            1         'benign'  
     5         1          2            1       2             1              3            1            1         'benign'  
     4         1          1            1       2             1              2            1            1         'benign'  
     6         1          1            1       2             1              2            1            1         'benign'  
     5         1          1            1       2             2              2            1            1         'benign'  
     3         1          1            1       2             1              1            1            1         'benign'  
     5         3          1            1       2             1              1            1            1         'benign'  
     4         1          1            1       2             1              2            1            1         'benign'  
     2         1          3            2       2             1              2            1            1         'benign'  
     5         1          1            1       2             1              2            1            1         'benign'  
     6        10         10           10       4            10              7           10            1         'malign'  
     2         1          1            1       1             1              1            1            1         'benign'  
     3         1          1            1       1             1              1            1            1         'benign'  
     7         8          3            7       4             5              7            8            2         'malign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     3         2          2            2       2             1              4            2            1         'benign'  
     4         4          2            1       2             5              2            1            2         'benign'  
     3         1          1            1       2             1              1            1            1         'benign'  
     4         3          1            1       2             1              4            8            1         'benign'  
     5         2          2            2       1             1              2            1            1         'benign'  
     5         1          1            3       2             1              1            1            1         'benign'  
     2         1          1            1       2             1              2            1            1         'benign'  
     5         1          1            1       2             1              2            1            1         'benign'  
     5         1          1            1       2             1              3            1            1         'benign'  
     5         1          1            1       2             1              3            1            1         'benign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     4         1          1            1       2             1              3            2            1         'benign'  
     5         7         10           10       5            10             10           10            1         'malign'  
     3         1          2            1       2             1              3            1            1         'benign'  
     4         1          1            1       2             3              2            1            1         'benign'  
     8         4          4            1       6            10              2            5            2         'malign'  
    10        10          8           10       6             5             10            3            1         'malign'  
     8        10          4            4       8            10              8            2            1         'malign'  
     7         6         10            5       3            10              9           10            2         'malign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              2            1            1         'benign'  
    10         9          7            3       4             2              7            7            1         'malign'  
     5         1          2            1       2             1              3            1            1         'benign'  
     5         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              3            1            1         'benign'  
     5         1          2            1       2             1              2            1            1         'benign'  
     5         7         10            6       5            10              7            5            1         'malign'  
     6        10          5            5       4            10              6           10            1         'malign'  
     3         1          1            1       2             1              1            1            1         'benign'  
     5         1          1            6       3             1              1            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     8        10         10           10       6            10             10           10            1         'malign'  
     5         1          1            1       2             1              2            2            1         'benign'  
     9         8          8            9       6             3              4            1            1         'malign'  
     5         1          1            1       2             1              1            1            1         'benign'  
     4        10          8            5       4             1             10            1            1         'malign'  
     2         5          7            6       4            10              7            6            1         'malign'  
    10         3          4            5       3            10              4            1            1         'malign'  
     5         1          2            1       2             1              1            1            1         'benign'  
     4         8          6            3       4            10              7            1            1         'malign'  
     5         1          1            1       2             1              2            1            1         'benign'  
     4         1          2            1       2             1              2            1            1         'benign'  
     5         1          3            1       2             1              3            1            1         'benign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     5         2          4            1       1             1              1            1            1         'benign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       1             1              2            1            1         'benign'  
     4         1          1            1       2             1              2            1            1         'benign'  
     5         4          6            8       4             1              8           10            1         'malign'  
     5         3          2            8       5            10              8            1            2         'malign'  
    10         5         10            3       5             8              7            8            3         'malign'  
     4         1          1            2       2             1              1            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     5        10         10           10      10            10             10            1            1         'malign'  
     5         1          1            1       2             1              1            1            1         'benign'  
    10         4          3           10       3            10              7            1            2         'malign'  
     5        10         10           10       5             2              8            5            1         'malign'  
     8        10         10           10       6            10             10           10           10         'malign'  
     2         3          1            1       2             1              2            1            1         'benign'  
     2         1          1            1       1             1              2            1            1         'benign'  
     4         1          3            1       2             1              2            1            1         'benign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       1             0              1            1            1         'benign'  
     4         1          1            1       2             1              2            1            1         'benign'  
     5         1          1            1       2             1              2            1            1         'benign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     6         3          3            3       3             2              6            1            1         'benign'  
     7         1          2            3       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     5         1          1            2       1             1              2            1            1         'benign'  
     3         1          3            1       3             4              1            1            1         'benign'  
     4         6          6            5       7             6              7            7            3         'malign'  
     2         1          1            1       2             5              1            1            1         'benign'  
     2         1          1            1       2             1              1            1            1         'benign'  
     4         1          1            1       2             1              1            1            1         'benign'  
     6         2          3            1       2             1              1            1            1         'benign'  
     5         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     8         7          4            4       5             3              5           10            1         'malign'  
     3         1          1            1       2             1              1            1            1         'benign'  
     3         1          4            1       2             1              1            1            1         'benign'  
    10        10          7            8       7             1             10           10            3         'malign'  
     4         2          4            3       2             2              2            1            1         'benign'  
     4         1          1            1       2             1              1            1            1         'benign'  
     5         1          1            3       2             1              1            1            1         'benign'  
     4         1          1            3       2             1              1            1            1         'benign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     2         1          1            1       2             1              1            1            1         'benign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     1         2          2            1       2             1              1            1            1         'benign'  
     1         1          1            3       2             1              1            1            1         'benign'  
     5        10         10           10      10             2             10           10           10         'malign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     3         1          1            2       3             4              1            1            1         'benign'  
     1         2          1            3       2             1              2            1            1         'benign'  
     5         1          1            1       2             1              2            2            1         'benign'  
     4         1          1            1       2             1              2            1            1         'benign'  
     3         1          1            1       2             1              3            1            1         'benign'  
     3         1          1            1       2             1              2            1            1         'benign'  
     5         1          1            1       2             1              2            1            1         'benign'  
     5         4          5            1       8             1              3            6            1         'benign'  
     7         8          8            7       3            10              7            2            3         'malign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     4         1          1            1       2             1              3            1            1         'benign'  
     1         1          3            1       2             1              2            1            1         'benign'  
     1         1          3            1       2             1              2            1            1         'benign'  
     3         1          1            3       2             1              2            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     5         2          2            2       2             1              1            1            2         'benign'  
     3         1          1            1       2             1              3            1            1         'benign'  
     5         7          4            1       6             1              7           10            3         'malign'  
     5        10         10            8       5             5              7           10            1         'malign'  
     3        10          7            8       5             8              7            4            1         'malign'  
     3         2          1            2       2             1              3            1            1         'benign'  
     2         1          1            1       2             1              3            1            1         'benign'  
     5         3          2            1       3             1              1            1            1         'benign'  
     1         1          1            1       2             1              2            1            1         'benign'  
     4         1          4            1       2             1              1            1            1         'benign'  
     1         1          2            1       2             1              2            1            1         'benign'  
     5         1          1            1       2             1              1            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     2         1          1            1       2             1              1            1            1         'benign'  
    10        10         10           10       5            10             10           10            7         'malign'  
     5        10         10           10       4            10              5            6            3         'malign'  
     5         1          1            1       2             1              3            2            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     1         1          1            1       2             1              1            1            1         'benign'  
     3         1          1            1       2             1              2            3            1         'benign'  
     4         1          1            1       2             1              1            1            1         'benign'  
     1         1          1            1       2             1              1            1            8         'benign'  
     1         1          1            3       2             1              1            1            1         'benign'  
     5        10         10            5       4             5              4            4            1         'malign'  
     3         1          1            1       2             1              1            1            1         'benign'  
     3         1          1            1       2             1              2            1            2         'benign'  
     3         1          1            1       3             2              1            1            1         'benign'  
     2         1          1            1       2             1              1            1            1         'benign'  
     5        10         10            3       7             3              8           10            2         'malign'  
     4         8          6            4       3             4             10            6            1         'malign'  
     4         8          8            5       4             5             10            4            1         'malign'


• Applied to another dataset (Indian Liver Patient) : 
Age     Gender     TotalBilirubin     DB     AAP     Sgpt    Sgot    TP     ALB    A_G     Selector
    ___    ________    ______________    ____    ____    ____    ____    ___    ___    ____    ________

    65     'Female'     0.7               0.1     187      16      18    6.8    3.3     0.9    1       
    62     'Male'      10.9               5.5     699      64     100    7.5    3.2    0.74    1       
    62     'Male'       7.3               4.1     490      60      68      7    3.3    0.89    1       
    58     'Male'         1               0.4     182      14      20    6.8    3.4       1    1       
    72     'Male'       3.9                 2     195      27      59    7.3    2.4     0.4    1       
    46     'Male'       1.8               0.7     208      19      14    7.6    4.4     1.3    1       
    26     'Female'     0.9               0.2     154      16      12      7    3.5       1    1       
    29     'Female'     0.9               0.3     202      14      11    6.7    3.6     1.1    1       
    17     'Male'       0.9               0.3     202      22      19    7.4    4.1     1.2    2       
    55     'Male'       0.7               0.2     290      53      58    6.8    3.4       1    1       
    57     'Male'       0.6               0.1     210      51      59    5.9    2.7     0.8    1       
    72     'Male'       2.7               1.3     260      31      56    7.4      3     0.6    1       
    64     'Male'       0.9               0.3     310      61      58      7    3.4     0.9    2       
    74     'Female'     1.1               0.4     214      22      30    8.1    4.1       1    1       
    61     'Male'       0.7               0.2     145      53      41    5.8    2.7    0.87    1       
    25     'Male'       0.6               0.1     183      91      53    5.5    2.3     0.7    2       
    38     'Male'       1.8               0.8     342     168     441    7.6    4.4     1.3    1       
    33     'Male'       1.6               0.5     165      15      23    7.3    3.5    0.92    2       
    40     'Female'     0.9               0.3     293     232     245    6.8    3.1     0.8    1       
    40     'Female'     0.9               0.3     293     232     245    6.8    3.1     0.8    1       
    51     'Male'       2.2                 1     610      17      28    7.3    2.6    0.55    1       
    51     'Male'       2.9               1.3     482      22      34      7    2.4     0.5    1       
    62     'Male'       6.8                 3     542     116      66    6.4    3.1     0.9    1       
    40     'Male'       1.9                 1     231      16      55    4.3    1.6     0.6    1       
    63     'Male'       0.9               0.2     194      52      45      6    3.9    1.85    2       
    34     'Male'       4.1                 2     289     875     731      5    2.7     1.1    1       
    34     'Male'       4.1                 2     289     875     731      5    2.7     1.1    1       
    34     'Male'       6.2                 3     240    1680     850    7.2      4     1.2    1       
    20     'Male'       1.1               0.5     128      20      30    3.9    1.9    0.95    2       
    84     'Female'     0.7               0.2     188      13      21      6    3.2     1.1    2       
    57     'Male'         4               1.9     190      45     111    5.2    1.5     0.4    1       
    52     'Male'       0.9               0.2     156      35      44    4.9    2.9     1.4    1       
    57     'Male'         1               0.3     187      19      23    5.2    2.9     1.2    2       
    38     'Female'     2.6               1.2     410      59      57    5.6      3     0.8    2       
    38     'Female'     2.6               1.2     410      59      57    5.6      3     0.8    2       
    30     'Male'       1.3               0.4     482     102      80    6.9    3.3     0.9    1       
    17     'Female'     0.7               0.2     145      18      36    7.2    3.9    1.18    2       
    46     'Female'    14.2               7.8     374      38      77    4.3      2     0.8    1       
    48     'Male'       1.4               0.6     263      38      66    5.8    2.2    0.61    1       
    47     'Male'       2.7               1.3     275     123      73    6.2    3.3     1.1    1       
    45     'Male'       2.4               1.1     168      33      50    5.1    2.6       1    1       
    62     'Male'       0.6               0.1     160      42     110    4.9    2.6     1.1    2       
    42     'Male'       6.8               3.2     630      25      47    6.1    2.3     0.6    2       
    50     'Male'       2.6               1.2     415     407     576    6.4    3.2       1    1       
    85     'Female'       1               0.3     208      17      15      7    3.6       1    2       
    35     'Male'       1.8               0.6     275      48     178    6.5    3.2     0.9    2       
    21     'Male'       3.9               1.8     150      36      27    6.8    3.9    1.34    1       
    40     'Male'       1.1               0.3     230    1630     960    4.9    2.8     1.3    1       
    32     'Female'     0.6               0.1     176      39      28      6      3       1    1       
    55     'Male'      18.4               8.8     206      64     178    6.2    1.8     0.4    1       
    45     'Female'     0.7               0.2     170      21      14    5.7    2.5     0.7    1       
    34     'Female'     0.6               0.1     161      15      19    6.6    3.4       1    1       
    38     'Male'       3.1               1.6     253      80     406    6.8    3.9     1.3    1       
    38     'Male'       1.1               0.3     198      86     150    6.3    3.5     1.2    1       
    42     'Male'       8.9               4.5     272      31      61    5.8      2     0.5    1       
    42     'Male'       8.9               4.5     272      31      61    5.8      2     0.5    1       
    33     'Male'       0.8               0.2     198      26      23      8      4       1    2       
    48     'Female'     0.9               0.2     175      24      54    5.5    2.7     0.9    2       
    51     'Male'       0.8               0.2     367      42      18    5.2      2     0.6    1       
    64     'Male'       1.1               0.5     145      20      24    5.5    3.2    1.39    2       
    31     'Female'     0.8               0.2     158      21      16      6      3       1    1       
    58     'Male'         1               0.5     158      37      43    7.2    3.6       1    1       
    58     'Male'         1               0.5     158      37      43    7.2    3.6       1    1       
    57     'Male'       0.7               0.2     208      35      97    5.1    2.1     0.7    1       
    57     'Male'       1.3               0.4     259      40      86    6.5    2.5     0.6    1       
    57     'Male'       1.4               0.7     470      62      88    5.6    2.5     0.8    1       
    54     'Male'       2.2               1.2     195      55      95      6    3.7     1.6    1       
    37     'Male'       1.8               0.8     215      53      58    6.4    3.8     1.4    1       
    66     'Male'       0.7               0.2     239      27      26    6.3    3.7     1.4    1       
    60     'Male'       0.8               0.2     215      24      17    6.3      3     0.9    2       
    19     'Female'     0.7               0.2     186     166     397    5.5      3     1.2    1       
    75     'Female'     0.8               0.2     188      20      29    4.4    1.8     0.6    1       
    75     'Female'     0.8               0.2     205      27      24    4.4      2     0.8    1       
    52     'Male'       0.6               0.1     171      22      16    6.6    3.6     1.2    1       
    68     'Male'       0.7               0.1     145      20      22    5.8    2.9       1    1       
    29     'Female'     0.7               0.1     162      52      41    5.2    2.5     0.9    2       
    31     'Male'       0.9               0.2     518     189      17    5.3    2.3     0.7    1       
    68     'Female'     0.6               0.1    1620      95     127    4.6    2.1     0.8    1       
    70     'Male'       1.4               0.6     146      12      24    6.2    3.8    1.58    2       
    58     'Female'     2.8               1.3     670      48      79    4.7    1.6     0.5    1       
    58     'Female'     2.4               1.1     915      60     142    4.7    1.8     0.6    1       
    29     'Male'         1               0.3      75      25      26    5.1    2.9     1.3    1       
    49     'Male'       0.7               0.1     148      14      12    5.4    2.8       1    2       
    33     'Male'         2                 1     258     194     152    5.4      3    1.25    1       
    32     'Male'       0.6               0.1     237      45      31    7.5    4.3    1.34    1       
    14     'Male'       1.4               0.5     269      58      45    6.7    3.9     1.4    1       
    13     'Male'       0.6               0.1     320      28      56    7.2    3.6       1    2       
    58     'Male'       0.8               0.2     298      33      59    6.2    3.1       1    1       
    18     'Male'       0.6               0.2     538      33      34    7.5    3.2     0.7    1       
    60     'Male'         4               1.9     238     119     350    7.1    3.3     0.8    1       
    60     'Male'       5.7               2.8     214     412     850    7.3    3.2    0.78    1       
    60     'Male'       6.8               3.2     308     404     794    6.8      3     0.7    1       
    60     'Male'       8.6                 4     298     412     850    7.4      3     0.6    1       
    60     'Male'       5.8               2.7     204     220     400      7      3     0.7    1       
    60     'Male'       5.2               2.4     168     126     202    6.8    2.9     0.7    1       
    75     'Male'       0.9               0.2     282      25      23    4.4    2.2       1    1       
    39     'Male'       3.8               1.5     298     102     630    7.1    3.3     0.8    1       
    39     'Male'       6.6                 3     215     190     950      4    1.7     0.7    1       
    18     'Male'       0.6               0.1     265      97     161    5.9    3.1     1.1    1       
    18     'Male'       0.7               0.1     312     308     405    6.9    3.7     1.1    1       
    27     'Male'       0.6               0.2     161      27      28    3.7    1.6    0.76    2       
    27     'Male'       0.7               0.2     243      21      23    5.3    2.3     0.7    2       
    17     'Male'       0.9               0.2     224      36      45    6.9    4.2    1.55    1       
    55     'Female'     0.8               0.2     225      14      23    6.1    3.3     1.2    2       
    63     'Male'       0.5               0.1     170      21      28    5.5    2.5     0.8    1       
    36     'Male'       5.3               2.3     145      32      92    5.1    2.6       1    2       
    36     'Male'       5.3               2.3     145      32      92    5.1    2.6       1    2       
    36     'Male'       0.8               0.2     158      29      39      6    2.2     0.5    2       
    36     'Male'       0.8               0.2     158      29      39      6    2.2     0.5    2       
    36     'Male'       0.9               0.1     486      25      34    5.9    2.8     0.9    2       
    24     'Female'     0.7               0.2     188      11      10    5.5    2.3    0.71    2       
    48     'Male'       3.2               1.6     257      33     116    5.7    2.2    0.62    1       
    27     'Male'       1.2               0.4     179      63      39    6.1    3.3     1.1    2       
    74     'Male'       0.6               0.1     272      24      98      5      2     0.6    1       
    50     'Male'       5.8                 3     661     181     285    5.7    2.3    0.67    2       
    50     'Male'       7.3               3.6    1580      88      64    5.6    2.3     0.6    2       
    48     'Male'       0.7               0.1    1630      74     149    5.3      2     0.6    1       
    32     'Male'      12.7               6.2     194    2000    2946    5.7    3.3     1.3    1       
    32     'Male'      15.9                 7     280    1350    1600    5.6    2.8       1    1       
    32     'Male'        18               8.2     298    1250    1050    5.4    2.6     0.9    1       
    32     'Male'        23              11.3     300     482     275    7.1    3.5     0.9    1       
    32     'Male'      22.7              10.2     290     322     113    6.6    2.8     0.7    1       
    58     'Male'       1.7               0.8     188      60      84    5.9    3.5     1.4    2       
    64     'Female'     0.8               0.2     178      17      18    6.3    3.1     0.9    1       
    28     'Male'       0.6               0.1     177      36      29    6.9    4.1     1.4    2       
    60     'Male'       1.8               0.5     201      45      25    3.9    1.7     0.7    2       
    48     'Male'       5.8               2.5     802     133      88      6    2.8     0.8    1       
    64     'Male'         3               1.4     248      46      40    6.5    3.2     0.9    1       
    58     'Female'     1.7               0.8    1896      61      83      8    3.9    0.95    1       
    45     'Male'       2.8               1.7     263      57      65    5.1    2.3     0.8    1       
    45     'Male'       3.2               1.4     512      50      58      6    2.7     0.8    1       
    70     'Female'     0.7               0.2     237      18      28    5.8    2.5    0.75    2       
    18     'Female'     0.8               0.2     199      34      31    6.5    3.5    1.16    2       
    53     'Male'       0.9               0.4     238      17      14    6.6    2.9     0.8    1       
    18     'Male'       1.8               0.7     178      35      36    6.8    3.6     1.1    1       
    66     'Male'      11.3               5.6    1110    1250    4929      7    2.4     0.5    1       
    46     'Female'     4.7               2.2     310      62      90    6.4    2.5     0.6    1       
    18     'Male'       0.8               0.2     282      72     140    5.5    2.5     0.8    1       
    18     'Male'       0.8               0.2     282      72     140    5.5    2.5     0.8    1       
    15     'Male'       0.8               0.2     380      25      66    6.1    3.7     1.5    1       
    60     'Male'       0.6               0.1     186      20      21    6.2    3.3     1.1    2       
    66     'Female'     4.2               2.1     159      15      30    7.1    2.2     0.4    1       
    30     'Male'       1.6               0.4     332      84     139    5.6    2.7     0.9    1       
    30     'Male'       1.6               0.4     332      84     139    5.6    2.7     0.9    1       
    45     'Female'     3.5               1.5     189      63      87    5.6    2.9       1    1       
    65     'Male'       0.8               0.2     201      18      22    5.4    2.9     1.1    2       
    66     'Female'     2.9               1.3     168      21      38    5.5    1.8     0.4    1       
    65     'Male'       0.7               0.1     392      20      30    5.3    2.8     1.1    1       
    50     'Male'       0.9               0.2     202      20      26    7.2    4.5    1.66    1       
    60     'Male'       0.8               0.2     286      21      27    7.1      4     1.2    1       
    56     'Male'       1.1               0.5     180      30      42    6.9    3.8     1.2    2       
    50     'Male'       1.6               0.8     218      18      20    5.9    2.9    0.96    1       
    46     'Female'     0.8               0.2     182      20      40      6    2.9     0.9    1       
    52     'Male'       0.6               0.1     178      26      27    6.5    3.6     1.2    2       
    34     'Male'       5.9               2.5     290      45     233    5.6    2.7     0.9    1       
    34     'Male'       8.7                 4     298      58     138    5.8    2.4     0.7    1       
    32     'Male'       0.9               0.3     462      70      82    6.2    3.1       1    1       
    72     'Male'       0.7               0.1     196      20      35    5.8      2     0.5    1       
    72     'Male'       0.7               0.1     196      20      35    5.8      2     0.5    1       
    50     'Male'       1.2               0.4     282      36      32    7.2    3.9     1.1    1       
    60     'Male'        11               4.9     750     140     350    5.5    2.1     0.6    1       
    60     'Male'      11.5                 5    1050      99     187    6.2    2.8     0.8    1       
    60     'Male'       5.8               2.7     599      43      66    5.4    1.8     0.5    1       
    39     'Male'       1.9               0.9     180      42      62    7.4    4.3    1.38    1       
    39     'Male'       1.9               0.9     180      42      62    7.4    4.3    1.38    1       
    48     'Male'       4.5               2.3     282      13      74      7    2.4    0.52    1       
    55     'Male'        75               3.6     332      40      66    6.2    2.5     0.6    1       
    47     'Female'       3               1.5     292      64      67    5.6    1.8    0.47    1       
    60     'Male'      22.8              12.6     962      53      41    6.9    3.3     0.9    1       
    60     'Male'       8.9                 4     950      33      32    6.8    3.1     0.8    1       
    72     'Male'       1.7               0.8     200      28      37    6.2      3    0.93    1       
    44     'Female'     1.9               0.6     298     378     602    6.6    3.3       1    1       
    55     'Male'      14.1               7.6     750      35      63      5    1.6    0.47    1       
    31     'Male'       0.6               0.1     175      48      34      6    3.7     1.6    1       
    31     'Male'       0.6               0.1     175      48      34      6    3.7     1.6    1       
    31     'Male'       0.8               0.2     198      43      31    7.3      4     1.2    1       
    55     'Male'       0.8               0.2     482     112      99    5.7    2.6     0.8    1       
    75     'Male'      14.8                 9    1020      71      42    5.3    2.2     0.7    1       
    75     'Male'      10.6                 5     562      37      29    5.1    1.8     0.5    1       
    75     'Male'         8               4.6     386      30      25    5.5    1.8    0.48    1       
    75     'Male'       2.8               1.3     250      23      29    2.7    0.9     0.5    1       
    75     'Male'       2.9               1.3     218      33      37      3    1.5       1    1       
    65     'Male'       1.9               0.8     170      36      43    3.8    1.4    0.58    2       
    40     'Male'       0.6               0.1     171      20      17    5.4    2.5     0.8    1       
    64     'Male'       1.1               0.4     201      18      19    6.9    4.1     1.4    1       
    38     'Male'       1.5               0.4     298      60     103      6      3       1    2       
    60     'Male'       3.2               1.8     750      79     145    7.8    3.2    0.69    1       
    60     'Male'       2.1                 1     191     114     247      4    1.6     0.6    1       
    60     'Male'       1.9               0.8     614      42      38    4.5    1.8     0.6    1       
    48     'Female'     0.8               0.2     218      32      28    5.2    2.5     0.9    2       
    60     'Male'       6.3               3.2     314     118     114    6.6    3.7    1.27    1       
    60     'Male'       5.8                 3     257     107     104    6.6    3.5    1.12    1       
    60     'Male'       2.3               0.6     272      79      51    6.6    3.5     1.1    1       
    49     'Male'       1.3               0.4     206      30      25      6    3.1    1.06    2       
    49     'Male'         2               0.6     209      48      32    5.7      3     1.1    2       
    60     'Male'       2.4                 1    1124      30      54    5.2    1.9     0.5    1       
    60     'Male'         2               1.1     664      52     104      6    2.1    0.53    1       
    26     'Female'     0.6               0.2     142      12      32    5.7    2.4    0.75    1       
    41     'Male'       0.9               0.2     169      22      18    6.1      3     0.9    2       
     7     'Female'    27.2              11.8    1420     790    1050    6.1      2     0.4    1       
    49     'Male'       0.6               0.1     218      50      53      5    2.4     0.9    1       
    49     'Male'       0.6               0.1     218      50      53      5    2.4     0.9    1       
    38     'Female'     0.8               0.2     145      19      23    6.1    3.1    1.03    2       
    21     'Male'         1               0.3     142      27      21    6.4    3.5     1.2    2       
    21     'Male'       0.7               0.2     135      27      26    6.4    3.3       1    2       
    45     'Male'       2.5               1.2     163      28      22    7.6      4     1.1    1       
    40     'Male'       3.6               1.8     285      50      60      7    2.9     0.7    1       
    40     'Male'       3.9               1.7     350     950    1500    6.7    3.8     1.3    1       
    70     'Female'     0.9               0.3     220      53      95    6.1    2.8    0.68    1       
    45     'Female'     0.9               0.3     189      23      33    6.6    3.9     NaN    1       
    28     'Male'       0.8               0.3     190      20      14    4.1    2.4     1.4    1       
    42     'Male'       2.7               1.3     219      60     180      7    3.2     0.8    1       
    22     'Male'       2.7                 1     160      82     127    5.5    3.1     1.2    2       
     8     'Female'     0.9               0.2     401      25      58    7.5    3.4     0.8    1       
    38     'Male'       1.7                 1     180      18      34    7.2    3.6       1    1       
    66     'Male'       0.6               0.2     100      17     148      5    3.3     1.9    2       
    55     'Male'       0.9               0.2     116      36      16    6.2    3.2       1    2       
    49     'Male'       1.1               0.5     159      30      31      7    4.3     1.5    1       
     6     'Male'       0.6               0.1     289      38      30    4.8      2     0.7    2       
    37     'Male'       0.8               0.2     125      41      39    6.4    3.4     1.1    1       
    37     'Male'       0.8               0.2     147      27      46      5    2.5       1    1       
    47     'Male'       0.9               0.2     192      38      24    7.3    4.3     1.4    1       
    47     'Male'       0.9               0.2     265      40      28      8      4       1    1       
    50     'Male'       1.1               0.3     175      20      19    7.1    4.5     1.7    2       
    70     'Male'       1.7               0.5     400      56      44    5.7    3.1     1.1    1       
    26     'Male'       0.6               0.2     120      45      51    7.9      4       1    1       
    26     'Male'       1.3               0.4     173      38      62      8      4       1    1       
    68     'Female'     0.7               0.2     186      18      15    6.4    3.8     1.4    1       
    65     'Female'       1               0.3     202      26      13    5.3    2.6     0.9    2       
    46     'Male'       0.6               0.2     290      26      21      6      3       1    1       
    61     'Male'       1.5               0.6     196      61      85    6.7    3.8     1.3    2       
    61     'Male'       0.8               0.1     282      85     231    8.5    4.3       1    1       
    50     'Male'       2.7               1.6     157     149     156    7.9    3.1     0.6    1       
    33     'Male'         2               1.4    2110      48      89    6.2      3     0.9    1       
    40     'Female'     0.9               0.2     285      32      27    7.7    3.5     0.8    1       
    60     'Male'       1.5               0.6     360     230     298    4.5      2     0.8    1       
    22     'Male'       0.8               0.2     300      57      40    7.9    3.8     0.9    2       
    35     'Female'     0.9               0.3     158      20      16      8      4       1    1       
    35     'Female'     0.9               0.2     190      40      35    7.3    4.7     1.8    2       
    40     'Male'       0.9               0.3     196      69      48    6.8    3.1     0.8    1       
    48     'Male'       0.7               0.2     165      32      30      8      4       1    2       
    51     'Male'       0.8               0.2     230      24      46    6.5    3.1     NaN    1       
    29     'Female'     0.8               0.2     205      30      23    8.2    4.1       1    1       
    28     'Female'     0.9               0.2     316      25      23    8.5    5.5     1.8    1       
    54     'Male'       0.8               0.2     218      20      19    6.3    2.5     0.6    1       
    54     'Male'       0.9               0.2     290      15      18    6.1    2.8     0.8    1       
    55     'Male'       1.8                 9     272      22      79    6.1    2.7     0.7    1       
    55     'Male'       0.9               0.2     190      25      28    5.9    2.7     0.8    1       
    40     'Male'       0.7               0.1     202      37      29      5    2.6       1    1       
    33     'Male'       1.2               0.3     498      28      25      7      3     0.7    1       
    33     'Male'       2.1               1.3     480      38      22    6.5      3     0.8    1       
    33     'Male'       0.9               0.8     680      37      40    5.9    2.6     0.8    1       
    65     'Male'       1.1               0.3     258      48      40      7    3.9     1.2    2       
    35     'Female'     0.6               0.2     180      12      15    5.2    2.7     NaN    2       
    38     'Female'     0.7               0.1     152      90      21    7.1    4.2     1.4    2       
    38     'Male'       1.7               0.7     859      89      48      6      3       1    1       
    50     'Male'       0.9               0.3     901      23      17    6.2    3.5     1.2    1       
    44     'Male'       0.8               0.2     335     148      86    5.6      3     1.1    1       
    36     'Male'       0.8               0.2     182      31      34    6.4    3.8     1.4    2       
    42     'Male'      30.5              14.2     285      65     130    5.2    2.1     0.6    1       
    42     'Male'      16.4               8.9     245      56      87    5.4      2     0.5    1       
    33     'Male'       1.5                 7     505     205     140    7.5    3.9       1    1       
    18     'Male'       0.8               0.2     228      55      54    6.9      4     1.3    1       
    38     'Female'     0.8               0.2     185      25      21      7      3     0.7    1       
    38     'Male'       0.8               0.2     247      55      92    7.4    4.3    1.38    2       
     4     'Male'       0.9               0.2     348      30      34      8      4       1    2       
    62     'Male'       1.2               0.4     195      38      54    6.3    3.8     1.5    1       
    43     'Female'     0.9               0.3     140      12      29    7.4    3.5     1.8    1       
    40     'Male'      14.5               6.4     358      50      75    5.7    2.1     0.5    1       
    26     'Male'       0.6               0.1     110      15      20    2.8    1.6     1.3    1       
    37     'Male'       0.7               0.2     235      96      54    9.5    4.9       1    1       
     4     'Male'       0.8               0.2     460     152     231    6.5    3.2     0.9    2       
    21     'Male'      18.5               9.5     380     390     500    8.2    4.1       1    1       
    30     'Male'       0.7               0.2     262      15      18    9.6    4.7     1.2    1       
    33     'Male'       1.8               0.8     196      25      22      8      4       1    1       
    26     'Male'       1.9               0.8     180      22      19    8.2    4.1       1    2       
    35     'Male'       0.9               0.2     190      25      20    6.4    3.6     1.2    2       
    60     'Male'         2               0.8     190      45      40      6    2.8     0.8    1       
    45     'Male'       2.2               0.8     209      25      20      8      4       1    1       
    48     'Female'       1               1.4     144      18      14    8.3    4.2       1    1       
    58     'Male'       0.8               0.2     123      56      48      6      3       1    1       
    50     'Male'       0.7               0.2     192      18      15    7.4    4.2     1.3    2       
    50     'Male'       0.7               0.2     188      12      14      7    3.4     0.9    1       
    18     'Male'       1.3               0.7     316      10      21      6    2.1     0.5    2       
    18     'Male'       0.9               0.3     300      30      48      8      4       1    1       
    13     'Male'       1.5               0.5     575      29      24    7.9    3.9     0.9    1       
    34     'Female'     0.8               0.2     192      15      12    8.6    4.7     1.2    1       
    43     'Male'       1.3               0.6     155      15      20      8      4       1    2       
    50     'Female'       1               0.5     239      16      39    7.5    3.7     0.9    1       
    57     'Male'       4.5               2.3     315     120     105      7      4     1.3    1       
    45     'Female'       1               0.3     250      48      44    8.6    4.3       1    1       
    60     'Male'       0.7               0.2     174      32      14    7.8    4.2     1.1    2       
    45     'Male'       0.6               0.2     245      22      24    7.1    3.4     0.9    1       
    23     'Male'       1.1               0.5     191      37      41    7.7    4.3     1.2    2       
    22     'Male'       2.4                 1     340      25      21    8.3    4.5     1.1    1       
    22     'Male'       0.6               0.2     202      78      41      8    3.9     0.9    1       
    74     'Female'     0.9               0.3     234      16      19    7.9      4       1    1       
    25     'Female'     0.9               0.3     159      24      25    6.9    4.4     1.7    2       
    31     'Female'     1.1               0.3     190      26      15    7.9    3.8     0.9    1       
    24     'Female'     0.9               0.2     195      40      35    7.4    4.1     1.2    2       
    58     'Male'       0.8               0.2     180      32      25    8.2    4.4     1.1    2       
    51     'Female'     0.9               0.2     280      21      30    6.7    3.2     0.8    1       
    50     'Female'     1.7               0.6     430      28      32    6.8    3.5       1    1       
    50     'Male'       0.7               0.2     206      18      17    8.4    4.2       1    2       
    55     'Female'     0.8               0.2     155      21      17    6.9    3.8     1.4    1       
    54     'Female'     1.4               0.7     195      36      16    7.9    3.7     0.9    2       
    48     'Male'       1.6                 1     588      74     113    7.3    2.4     0.4    1       
    30     'Male'       0.8               0.2     174      21      47    4.6    2.3       1    1       
    45     'Female'     0.8               0.2     165      22      18    8.2    4.1       1    1       
    48     'Female'     1.1               0.7     527     178     250      8    4.2     1.1    1       
    51     'Male'       0.8               0.2     175      48      22    8.1    4.6     1.3    1       
    54     'Female'    23.2              12.6     574      43      47    7.2    3.5     0.9    1       
    27     'Male'       1.3               0.6     106      25      54    8.5    4.8     NaN    2       
    30     'Female'     0.8               0.2     158      25      22    7.9    4.5     1.3    2       
    26     'Male'         2               0.9     195      24      65    7.8    4.3     1.2    1       
    22     'Male'       0.9               0.3     179      18      21    6.7    3.7     1.2    2       
    44     'Male'       0.9               0.2     182      29      82    7.1    3.7       1    2       
    35     'Male'       0.7               0.2     198      42      30    6.8    3.4       1    1       
    38     'Male'       3.7               2.2     216     179     232    7.8    4.5     1.3    1       
    14     'Male'       0.9               0.3     310      21      16    8.1    4.2       1    2       
    30     'Female'     0.7               0.2      63      31      27    5.8    3.4     1.4    1       
    30     'Female'     0.8               0.2     198      30      58    5.2    2.8     1.1    1       
    36     'Male'       1.7               0.5     205      36      34    7.1    3.9     1.2    1       
    12     'Male'       0.8               0.2     302      47      67    6.7    3.5     1.1    2       
    60     'Male'       2.6               1.2     171      42      37    5.4    2.7       1    1       
    42     'Male'       0.8               0.2     158      27      23    6.7    3.1     0.8    2       
    36     'Female'     1.2               0.4     358     160      90    8.3    4.4     1.1    2       
    24     'Male'       3.3               1.6     174      11      33    7.6    3.9       1    2       
    43     'Male'       0.8               0.2     192      29      20      6    2.9     0.9    2       
    21     'Male'       0.7               0.2     211      14      23    7.3    4.1     1.2    2       
    26     'Male'         2               0.9     157      54      68    6.1    2.7     0.8    1       
    26     'Male'       1.7               0.6     210      62      56    5.4    2.2     0.6    1       
    26     'Male'       7.1               3.3     258      80     113    6.2    2.9     0.8    1       
    36     'Female'     0.7               0.2     152      21      25    5.9    3.1     1.1    2       
    13     'Female'     0.7               0.2     350      17      24    7.4      4     1.1    1       
    13     'Female'     0.7               0.1     182      24      19    8.9    4.9     1.2    1       
    75     'Male'       6.7               3.6     458     198     143    6.2    3.2       1    1       
    75     'Male'       2.5               1.2     375      85      68    6.4    2.9     0.8    1       
    75     'Male'       1.8               0.8     405      79      50    6.1    2.9     0.9    1       
    75     'Male'       1.4               0.4     215      50      30    5.9    2.6     0.7    1       
    75     'Male'       0.9               0.2     206      44      33    6.2    2.9     0.8    1       
    36     'Female'     0.8               0.2     650      70     138    6.6    3.1     0.8    1       
    35     'Male'       0.8               0.2     198      36      32      7      4     1.3    2       
    70     'Male'       3.1               1.6     198      40      28    5.6      2     0.5    1       
    37     'Male'       0.8               0.2     195      60      40    8.2      5     1.5    2       
    60     'Male'       2.9               1.3     230      32      44    5.6      2     0.5    1       
    46     'Male'       0.6               0.2     115      14      11    6.9    3.4     0.9    1       
    38     'Male'       0.7               0.2     216     349     105      7    3.5       1    1       
    70     'Male'       1.3               0.4     358      19      14    6.1    2.8     0.8    1       
    49     'Female'     0.8               0.2     158      19      15    6.6    3.6     1.2    2       
    37     'Male'       1.8               0.8     145      62      58    5.7    2.9       1    1       
    37     'Male'       1.3               0.4     195      41      38    5.3    2.1     0.6    1       
    26     'Female'     0.7               0.2     144      36      33    8.2    4.3     1.1    1       
    48     'Female'     1.4               0.8     621     110     176    7.2    3.9     1.1    1       
    48     'Female'     0.8               0.2     150      25      23    7.5    3.9       1    1       
    19     'Male'       1.4               0.8     178      13      26      8    4.6     1.3    2       
    33     'Male'       0.7               0.2     256      21      30    8.5    3.9     0.8    1       
    33     'Male'       2.1               0.7     205      50      38    6.8      3     0.7    1       
    37     'Male'       0.7               0.2     176      28      34    5.6    2.6     0.8    1       
    69     'Female'     0.8               0.2     146      42      70    8.4    4.9     1.4    2       
    24     'Male'       0.7               0.2     218      47      26    6.6    3.3       1    1       
    65     'Female'     0.7               0.2     182      23      28    6.8    2.9     0.7    2       
    55     'Male'       1.1               0.3     215      21      15    6.2    2.9     0.8    2       
    42     'Female'     0.9               0.2     165      26      29    8.5    4.4       1    2       
    21     'Male'       0.8               0.2     183      33      57    6.8    3.5       1    2       
    40     'Male'       0.7               0.2     176      28      43    5.3    2.4     0.8    2       
    16     'Male'       0.7               0.2     418      28      35    7.2    4.1     1.3    2       
    60     'Male'       2.2                 1     271      45      52    6.1    2.9     0.9    2       
    42     'Female'     0.8               0.2     182      22      20    7.2    3.9     1.1    1       
    58     'Female'     0.8               0.2     130      24      25      7      4     1.3    1       
    54     'Female'    22.6              11.4     558      30      37    7.8    3.4     0.8    1       
    33     'Male'       0.8               0.2     135      30      29    7.2    4.4     1.5    2       
    48     'Male'       0.7               0.2     326      29      17    8.7    5.5     1.7    1       
    25     'Female'     0.7               0.1     140      32      25    7.6    4.3     1.3    2       
    56     'Female'     0.7               0.1     145      26      23      7      4     1.3    2       
    47     'Male'       3.5               1.6     206      32      31    6.8    3.4       1    1       
    33     'Male'       0.7               0.1     168      35      33      7    3.7     1.1    1       
    20     'Female'     0.6               0.2     202      12      13    6.1      3     0.9    2       
    50     'Female'     0.7               0.1     192      20      41    7.3    3.3     0.8    1       
    72     'Male'       0.7               0.2     185      16      22    7.3    3.7       1    2       
    50     'Male'       1.7               0.8     331      36      53    7.3    3.4     0.9    1       
    39     'Male'       0.6               0.2     188      28      43    8.1    3.3     0.6    1       
    58     'Female'     0.7               0.1     172      27      22    6.7    3.2     0.9    1       
    60     'Female'     1.4               0.7     159      10      12    4.9    2.5       1    2       
    34     'Male'       3.7               2.1     490     115      91    6.5    2.8     0.7    1       
    50     'Male'       0.8               0.2     152      29      30    7.4    4.1     1.3    1       
    38     'Male'       2.7               1.4     105      25      21    7.5    4.2     1.2    2       
    51     'Male'       0.8               0.2     160      34      20    6.9    3.7     1.1    1       
    46     'Male'       0.8               0.2     160      31      40    7.3    3.8     1.1    1       
    72     'Male'       0.6               0.1     102      31      35    6.3    3.2       1    1       
    72     'Male'       0.8               0.2     148      23      35      6      3       1    1       
    75     'Male'       0.9               0.2     162      25      20    6.9    3.7     1.1    1       
    41     'Male'       7.5               4.3     149      94      92    6.3    3.1     0.9    1       
    41     'Male'       2.7               1.3     580     142      68      8      4       1    1       
    48     'Female'       1               0.3     310      37      56    5.9    2.5     0.7    1       
    45     'Male'       0.8               0.2     140      24      20    6.3    3.2       1    2       
    74     'Male'         1               0.3     175      30      32    6.4    3.4     1.1    1       
    78     'Male'         1               0.3     152      28      70    6.3    3.1     0.9    1       
    38     'Male'       0.8               0.2     208      25      50    7.1    3.7       1    1       
    27     'Male'         1               0.2     205     137     145      6      3       1    1       
    66     'Female'     0.7               0.2     162      24      20    6.4    3.2       1    2       
    50     'Male'       7.3               3.7      92      44     236    6.8    1.6     0.3    1       
    42     'Female'     0.5               0.1     162     155     108    8.1      4     0.9    1       
    65     'Male'       0.7               0.2     199      19      22    6.3    3.6     1.3    2       
    22     'Male'       0.8               0.2     198      20      26    6.8    3.9     1.3    1       
    31     'Female'     0.8               0.2     215      15      21    7.6      4     1.1    1       
    45     'Male'       0.7               0.2     180      18      58    6.7    3.7     1.2    2       
    12     'Male'         1               0.2     719     157     108    7.2    3.7       1    1       
    48     'Male'       2.4               1.1     554     141      73    7.5    3.6     0.9    1       
    48     'Male'         5               2.6     555     284     190    6.5    3.3       1    1       
    18     'Male'       1.4               0.6     215     440     850      5    1.9     0.6    1       
    23     'Female'     2.3               0.8     509      28      44    6.9    2.9     0.7    2       
    65     'Male'       4.9               2.7     190      33      71    7.1    2.9     0.7    1       
    48     'Male'       0.7               0.2     208      15      30    4.6    2.1     0.8    2       
    65     'Male'       1.4               0.6     260      28      24    5.2    2.2     0.7    2       
    70     'Male'       1.3               0.3     690      93      40    3.6    2.7     0.7    1       
    70     'Male'       0.6               0.1     862      76     180    6.3    2.7    0.75    1       
    11     'Male'       0.7               0.1     592      26      29    7.1    4.2     1.4    2       
    50     'Male'       4.2               2.3     450      69      50      7      3     0.7    1       
    55     'Female'     8.2               3.9    1350      52      65    6.7    2.9     0.7    1       
    55     'Female'    10.9               5.1    1350      48      57    6.4    2.3     0.5    1       
    26     'Male'         1               0.3     163      48      71    7.1    3.7       1    2       
    41     'Male'       1.2               0.5     246      34      42    6.9    3.4    0.97    1       
    53     'Male'       1.6               0.9     178      44      59    6.5    3.9     1.5    2       
    32     'Female'     0.7               0.1     240      12      15      7      3     0.7    1       
    58     'Male'       0.4               0.1     100      59     126    4.3    2.5     1.4    1       
    45     'Male'       1.3               0.6     166      49      42    5.6    2.5     0.8    2       
    65     'Male'       0.9               0.2     170      33      66      7      3    0.75    1       
    52     'Female'     0.6               0.1     194      10      12    6.9    3.3     0.9    2       
    73     'Male'       1.9               0.7    1750     102     141    5.5      2     0.5    1       
    53     'Female'     0.7               0.1     182      20      33    4.8    1.9     0.6    1       
    47     'Female'     0.8               0.2     236      10      13    6.7    2.9    0.76    2       
    29     'Male'       0.7               0.2     165      55      87    7.5    4.6    1.58    1       
    41     'Female'     0.9               0.2     201      31      24    7.6    3.8       1    2       
    30     'Female'     0.7               0.2     194      32      36    7.5    3.6    0.92    2       
    17     'Female'     0.5               0.1     206      28      21    7.1    4.5     1.7    2       
    23     'Male'         1               0.3     212      41      80    6.2    3.1       1    1       
    35     'Male'       1.6               0.7     157      15      44    5.2    2.5     0.9    1       
    65     'Male'       0.8               0.2     162      30      90    3.8    1.4     0.5    1       
    42     'Female'     0.8               0.2     168      25      18    6.2    3.1       1    1       
    49     'Female'     0.8               0.2     198      23      20      7    4.3     1.5    1       
    42     'Female'     2.3               1.1     292      29      39    4.1    1.8     0.7    1       
    42     'Female'     7.4               3.6     298      52     102    4.6    1.9     0.7    1       
    42     'Female'     0.7               0.2     152      35      81    6.2    3.2    1.06    1       
    61     'Male'       0.8               0.2     163      18      19    6.3    2.8     0.8    2       
    17     'Male'       0.9               0.2     279      40      46    7.3      4     1.2    2       
    54     'Male'       0.8               0.2     181      35      20    5.5    2.7    0.96    1       
    45     'Female'    23.3              12.8    1550     425     511    7.7    3.5     0.8    1       
    48     'Female'     0.8               0.2     142      26      25      6    2.6     0.7    1       
    48     'Female'     0.9               0.2     173      26      27    6.2    3.1       1    1       
    65     'Male'       7.9               4.3     282      50      72      6      3       1    1       
    35     'Male'       0.8               0.2     279      20      25    7.2    3.2     0.8    1       
    58     'Male'       0.9               0.2    1100      25      36    7.1    3.5     0.9    1       
    46     'Male'       0.7               0.2     224      40      23    7.1      3     0.7    1       
    28     'Male'       0.6               0.2     159      15      16      7    3.5       1    2       
    21     'Female'     0.6               0.1     186      25      22    6.8    3.4       1    1       
    32     'Male'       0.7               0.2     189      22      43    7.4    3.1     0.7    2       
    61     'Male'       0.8               0.2     192      28      35    6.9    3.4     0.9    2       
    26     'Male'       6.8               3.2     140      37      19    3.6    0.9     0.3    1       
    65     'Male'       1.1               0.5     686      16      46    5.7    1.5    0.35    1       
    22     'Female'     2.2                 1     215     159      51    5.5    2.5     0.8    1       
    28     'Female'     0.8               0.2     309      55      23    6.8    4.1    1.51    1       
    38     'Male'       0.7               0.2     110      22      18    6.4    2.5    0.64    1       
    25     'Male'       0.8               0.1     130      23      42      8      4       1    1       
    45     'Female'     0.7               0.2     164      21      53    4.5    1.4    0.45    2       
    45     'Female'     0.6               0.1     270      23      42    5.1      2     0.5    2       
    28     'Female'     0.6               0.1     137      22      16    4.9    1.9     0.6    2       
    28     'Female'       1               0.3      90      18     108    6.8    3.1     0.8    2       
    66     'Male'         1               0.3     190      30      54    5.3    2.1     0.6    1       
    66     'Male'       0.8               0.2     165      22      32    4.4      2     0.8    1       
    66     'Male'       1.1               0.5     167      13      56    7.1    4.1    1.36    1       
    49     'Female'     0.6               0.1     185      17      26    6.6    2.9     0.7    2       
    42     'Male'       0.7               0.2     197      64      33    5.8    2.4     0.7    2       
    42     'Male'         1               0.3     154      38      21    6.8    3.9     1.3    2       
    35     'Male'         2               1.1     226      33     135      6    2.7     0.8    2       
    38     'Male'       2.2                 1     310     119      42    7.9    4.1       1    2       
    38     'Male'       0.9               0.3     310      15      25    5.5    2.7       1    1       
    55     'Male'       0.6               0.2     220      24      32    5.1    2.4    0.88    1       
    33     'Male'       7.1               3.7     196     622     497    6.9    3.6    1.09    1       
    33     'Male'       3.4               1.6     186     779     844    7.3    3.2     0.7    1       
     7     'Male'       0.5               0.1     352      28      51    7.9    4.2     1.1    2       
    45     'Male'       2.3               1.3     282     132     368    7.3      4     1.2    1       
    45     'Male'       1.1               0.4      92      91     188    7.2    3.8    1.11    1       
    30     'Male'       0.8               0.2     182      46      57    7.8    4.3     1.2    2       
    62     'Male'         5               2.1     103      18      40      5    2.1    1.72    1       
    22     'Female'     6.7               3.2     850     154     248    6.2    2.8     0.8    1       
    42     'Female'     0.8               0.2     195      18      15    6.7      3     0.8    1       
    32     'Male'       0.7               0.2     276     102     190      6    2.9    0.93    1       
    60     'Male'       0.7               0.2     171      31      26      7    3.5       1    2       
    65     'Male'       0.8               0.1     146      17      29    5.9    3.2    1.18    2       
    53     'Female'     0.8               0.2     193      96      57    6.7    3.6    1.16    1       
    27     'Male'         1               0.3     180      56     111    6.8    3.9    1.85    2       
    35     'Female'       1               0.3     805     133     103    7.9    3.3     0.7    1       
    65     'Male'       0.7               0.2     265      30      28    5.2    1.8    0.52    2       
    25     'Male'       0.7               0.2     185     196     401    6.5    3.9     1.5    1       
    32     'Male'       0.7               0.2     165      31      29    6.1      3    0.96    2       
    24     'Male'         1               0.2     189      52      31      8    4.8     1.5    1       
    67     'Male'       2.2               1.1     198      42      39    7.2      3     0.7    1       
    68     'Male'       1.8               0.5     151      18      22    6.5      4     1.6    1       
    55     'Male'       3.6               1.6     349      40      70    7.2    2.9     0.6    1       
    70     'Male'       2.7               1.2     365      62      55      6    2.4     0.6    1       
    36     'Male'       2.8               1.5     305      28      76    5.9    2.5     0.7    1       
    42     'Male'       0.8               0.2     127      29      30    4.9    2.7     1.2    1       
    53     'Male'      19.8              10.4     238      39     221    8.1    2.5     0.4    1       
    32     'Male'      30.5              17.1     218      39      79    5.5    2.7     0.9    1       
    32     'Male'      32.6              14.1     219      95     235    5.8    3.1     1.1    1       
    56     'Male'      17.7               8.8     239      43     185    5.6    2.4     0.7    1       
    50     'Male'       0.9               0.3     194     190      73    7.5    3.9       1    1       
    46     'Male'      18.4               8.5     450     119     230    7.5    3.3     0.7    1       
    46     'Male'        20                10     254     140     540    5.4      3     1.2    1       
    37     'Female'     0.8               0.2     205      31      36    9.2    4.6       1    2       
    45     'Male'       2.2               1.6     320      37      48    6.8    3.4       1    1       
    56     'Male'         1               0.3     195      22      28    5.8    2.6     0.8    2       
    69     'Male'       0.9               0.2     215      32      24    6.9      3     0.7    1       
    49     'Male'         1               0.3     230      48      58    8.4    4.2       1    1       
    49     'Male'       3.9               2.1     189      65     181    6.9      3     0.7    1       
    60     'Male'       0.9               0.3     168      16      24    6.7      3     0.8    1       
    28     'Male'       0.9               0.2     215      50      28      8      4       1    1       
    45     'Male'       2.9               1.4     210      74      68    7.2    3.6       1    1       
    35     'Male'      26.3              12.1     108     168     630    9.2      2     0.3    1       
    62     'Male'       1.8               0.9     224      69     155    8.6      4     0.8    1       
    55     'Male'       4.4               2.9     230      14      25    7.1    2.1     0.4    1       
    46     'Female'     0.8               0.2     185      24      15    7.9    3.7     0.8    1       
    50     'Male'       0.6               0.2     137      15      16    4.8    2.6     1.1    1       
    29     'Male'       0.8               0.2     156      12      15    6.8    3.7     1.1    2       
    53     'Female'     0.9               0.2     210      35      32      8    3.9     0.9    2       
    46     'Male'       9.4               5.2     268      21      63    6.4    2.8     0.8    1       
    40     'Male'       3.5               1.6     298      68     200    7.1    3.4     0.9    1       
    45     'Male'       1.7               0.8     315      12      38    6.3    2.1     0.5    1       
    55     'Male'       3.3               1.5     214      54     152    5.1    1.8     0.5    1       
    22     'Female'     1.1               0.3     138      14      21      7    3.8     1.1    2       
    40     'Male'      30.8              18.3     285     110     186    7.9    2.7     0.5    1       
    62     'Male'       0.7               0.2     162      12      17    8.2    3.2     0.6    2       
    46     'Female'     1.4               0.4     298     509     623    3.6      1     0.3    1       
    39     'Male'       1.6               0.8     230      88      74      8      4       1    2       
    60     'Male'      19.6               9.5     466      46      52    6.1      2     0.4    1       
    46     'Male'      15.8               7.2     227      67     220    6.9    2.6     0.6    1       
    10     'Female'     0.8               0.1     395      25      75    7.6    3.6     0.9    1       
    52     'Male'       1.8               0.8      97      85      78    6.4    2.7     0.7    1       
    65     'Female'     0.7               0.2     406      24      45    7.2    3.5     0.9    2       
    42     'Male'       0.8               0.2     114      21      23      7      3     0.7    2       
    42     'Male'       0.8               0.2     198      29      19    6.6      3     0.8    2       
    62     'Male'       0.7               0.2     173      46      47    7.3    4.1     1.2    2       
    40     'Male'       1.2               0.6     204      23      27    7.6      4     1.1    1       
    54     'Female'     5.5               3.2     350      67      42      7    3.2     0.8    1       
    45     'Female'     0.7               0.2     153      41      42    4.5    2.2     0.9    2       
    45     'Male'      20.2              11.7     188      47      32    5.4    2.3     0.7    1       
    50     'Female'    27.7              10.8     380      39     348    7.1    2.3     0.4    1       
    42     'Male'      11.1               6.1     214      60     186    6.9    2.8     2.8    1       
    40     'Female'     2.1                 1     768      74     141    7.8    4.9     1.6    1       
    46     'Male'       3.3               1.5     172      25      41    5.6    2.4     0.7    1       
    29     'Male'       1.2               0.4     160      20      22    6.2      3     0.9    2       
    45     'Male'       0.6               0.1     196      29      30    5.8    2.9       1    1       
    46     'Male'      10.2               4.2     232      58     140      7    2.7     0.6    1       
    73     'Male'       1.8               0.9     220      20      43    6.5      3     0.8    1       
    55     'Male'       0.8               0.2     290     139      87      7      3     0.7    1       
    51     'Male'       0.7               0.1     180      25      27    6.1    3.1       1    1       
    51     'Male'       2.9               1.2     189      80     125    6.2    3.1       1    1       
    51     'Male'         4               2.5     275     382     330    7.5      4     1.1    1       
    26     'Male'      42.8              19.7     390      75     138    7.5    2.6     0.5    1       
    66     'Male'      15.2               7.7     356     321     562    6.5    2.2     0.4    1       
    66     'Male'      16.6               7.6     315     233     384    6.9      2     0.4    1       
    66     'Male'      17.3               8.5     388     173     367    7.8    2.6     0.5    1       
    64     'Male'       1.4               0.5     298      31      83    7.2    2.6     0.5    1       
    38     'Female'     0.6               0.1     165      22      34    5.9    2.9     0.9    2       
    43     'Male'      22.5              11.8     143      22     143    6.6    2.1    0.46    1       
    50     'Female'       1               0.3     191      22      31    7.8      4       1    2       
    52     'Male'       2.7               1.4     251      20      40      6    1.7    0.39    1       
    20     'Female'    16.7               8.4     200      91     101    6.9    3.5    1.02    1       
    16     'Male'       7.7               4.1     268     213     168    7.1      4     1.2    1       
    16     'Male'       2.6               1.2     236     131      90    5.4    2.6     0.9    1       
    90     'Male'       1.1               0.3     215      46     134    6.9      3     0.7    1       
    32     'Male'      15.6               9.5     134      54     125    5.6      4     2.5    1       
    32     'Male'       3.7               1.6     612      50      88    6.2    1.9     0.4    1       
    32     'Male'      12.1                 6     515      48      92    6.6    2.4     0.5    1       
    32     'Male'        25              13.7     560      41      88    7.9    2.5     2.5    1       
    32     'Male'        15               8.2     289      58      80    5.3    2.2     0.7    1       
    32     'Male'      12.7               8.4     190      28      47    5.4    2.6     0.9    1       
    60     'Male'       0.5               0.1     500      20      34    5.9    1.6    0.37    2       
    40     'Male'       0.6               0.1      98      35      31      6    3.2     1.1    1       
    52     'Male'       0.8               0.2     245      48      49    6.4    3.2       1    1       
    31     'Male'       1.3               0.5     184      29      32    6.8    3.4       1    1       
    38     'Male'         1               0.3     216      21      24    7.3    4.4     1.5    2       
