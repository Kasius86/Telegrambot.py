a = 7
b = 10
c = 5
d = 6

print(' ', end='\t')
for i in range(c, d+1):
  print (i, end='\t')   for j in range(a, b+1):
        print ()
        print(j, end='')
        for j in range(a, b+1):
          for i in range(c, d+1):
            print(i*j)