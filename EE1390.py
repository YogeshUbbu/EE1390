import numpy as np
import matplotlib.pyplot as plt


#      Given lines:
#                  (1,-1)x=-1
#                  (7,-1)x=5
#       Given center: (-1,-2)

def intersection(N,P):
    return np.linalg.solve(N,P)

def otherPoint(A,B):
    return 2*B-A

def normal(AB):
    return np.matmul(AB,Slope_vec)

M=np.array([-1,-2])
n1=np.array([1,-1])                #given
n2=np.array([7,-1])
P=np.zeros(2)
P[0]=-1
P[1]=5
Slope_vec=np.array([-1,1])

N=np.vstack((n1,n2))
A=intersection(N,P)            #Point A

C=otherPoint(A,M)             #Point C


AB = np.vstack((A, M)).T
n3=normal(AB)                #finding the normal of perpendicular
P2=np.matmul(n3,M.T)

P[0]=P2
N=np.vstack((n3,n2))
B=intersection(N,P)
D=otherPoint(B,M)


len=20

lam_1 = np.linspace(0, 1, len)
a = np.zeros((2, len))
b = np.zeros((2, len))
c = np.zeros((2, len))
d = np.zeros((2, len))
e = np.zeros((2, len))
f = np.zeros((2, len))

for i in range(len):
    temp1 = A + lam_1[i] * (B - A)
    a[:, i] = temp1.T
    temp2 = B + lam_1[i] * (C - B)
    b[:, i] = temp2.T

    temp3 = C + lam_1[i] * (A - C)
    c[:, i] = temp3.T
    temp4 = D + lam_1[i] * (B - D)
    d[:, i] = temp4.T
    temp5 = D + lam_1[i] * (A - D)
    e[:, i] = temp5.T
    temp6 = C + lam_1[i] * (D - C)
    f[:, i] = temp6.T

plt.plot(a[0, :], a[1,:], label='$AB$')
plt.plot(b[0, :], b[1,:], label='$BC$')
plt.plot(c[0, :], c[1,:], label='$CA$')
plt.plot(d[0, :], d[1,:], label='$AD$')
plt.plot(e[0, :], e[1,:], label='$BD$')
plt.plot(f[0, :], f[1,:], label='$CD$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1), 'A(1,2)')
plt.plot(B[0], B[1],'o')
plt.text(B[0] * (1 + 0.2), B[1] * (1), 'B(-0.3,-2.6)')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 - 0.05), C[1] * (1 + 0.03), 'C(-3,-6)')
plt.plot(D[0], D[1], 'o')
plt.text(D[0] * (1 + 0.03), D[1] * (1 - 0.1), 'D(-2.3,1.3)')
plt.plot(M[0],M[1],'o')
plt.text(M[0] * (1 + 0.03), M[1] * (1 - 0.1), 'M(-1,-2)')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.show()

