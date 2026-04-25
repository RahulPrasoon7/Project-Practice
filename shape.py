n = 5

for i in range(n):
    for j in range(2*n - 1):
        # top arcs
        if (i == 0 and (j == 1 or j == 2 or j == 4 or j == 5 or j == 7 or j == 8)):
            print("*", end=" ")
        
        # second row connectors
        elif (i == 1 and (j == 0 or j == 3 or j == 4 or j == 5 or j == 6 or j == 9)):
            print("*", end=" ")
        
        # left & right borders
        elif (j == 0 or j == 2*n - 2):
            if i > 1:
                print("*", end=" ")
            else:
                print(" ", end=" ")
        
        # inner diagonals (V shape)
        elif (i + j == n + 1 or j - i == n - 2):
            print("*", end=" ")
        
        else:
            print(" ", end=" ")
    
    print()
        