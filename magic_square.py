def magic_square (n) :
    
    magicSquare = []
    for i in range(3):
        l = []
        for j in range(3):
            l.append(0)
        magicSquare.append(l)

    
    for i in range(3):  
        for j in range(3):
            print(magicSquare [i][j])
 