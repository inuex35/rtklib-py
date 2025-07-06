import gtsam
from gtsam import symbol_shorthand as S

X = S.X
V = S.V  
B = S.B
C = S.C

print("Epoch 7 symbols:")
print(f"X(7) = {X(7)} (pose)")
print(f"V(7) = {V(7)} (velocity)")
print(f"B(7) = {B(7)} (bias)")
print(f"C(7) = {C(7)} (clock)")

print("\nEpoch 8 symbols:")
print(f"X(8) = {X(8)}")
print(f"V(8) = {V(8)}")
print(f"B(8) = {B(8)}")
print(f"C(8) = {C(8)}")

print("\nSymbol causing error: 8646911284551352327")
print(f"This is X(7) = {X(7)}")

print("\nKeys in error log:")
print("7061644215716937736 =", B(8))
print("7133701809754865672 =", C(8))
print("8502796096475496456 =", V(8))
print("8646911284551352328 =", X(8))