import numpy as np

# Equivalent to `np.linalg.qr(a=a, mode='complete')`.
def gram_schmidt_qr(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  m, n = a.shape
  q = np.zeros(shape=(m, n), dtype=float)
  r = np.zeros(shape=(n, n), dtype=float)
  for i in range(n):
    vi = a[:, i].copy()
    for j in range(i):
      r[j, i] = np.dot(q[:, j], vi)
      vi -= r[j, i] * q[:, j]
    r[i, i] = np.linalg.norm(vi)  
    q[:, i] = vi / r[i, i]
  return q, r  


# Test the code
a = np.array([[12, -51, 4], 
              [6, 167, -68], 
              [-4, 24, -41]], dtype=float)
q, r = gram_schmidt_qr(a=a)
print("Q:\n", q)
print("R:\n", r)
print("QR:\n", np.dot(q, r))
