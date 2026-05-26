--- Attempting Simulated Annealing with Fixed k ---

--- Test 1a: Sum of L/9 functions (p=3, n=4) ---
Generating basis functions and their polynomial representations...
Found 19683 unique basis functions.

✅ SA Best representation found with fixed k=6:
  Final Cost: 3.8125e-14
  Final Error: 3.81e-14
  Basis functions used:
    - (Orig Idx 14952) q(y) = (2*y_0)/3 + (2*y_2)/3 + (2*y_0^2)/3 + (y_1^2)/3 + (y_0*y_1)/3 + (y_0*y_2)/3 + (y_1*y_2)/3
      Coefficient: 1.5000 + 0.8660j
    - (Orig Idx 17392) q(y) = (2*y_0)/3 + (y_1)/3 + (2*y_2)/3 + (y_1^2)/3 + (y_2^2)/3 + (2*y_0*y_1)/3 + (y_0*y_2)/3 + (2*y_1*y_2)/3
      Coefficient: 1.5000 + 0.8660j
    - (Orig Idx 40) q(y) = (y_0^2)/3 + (y_1^2)/3 + (y_2^2)/3 + (y_1*y_2)/3
      Coefficient: 1.5000 + -0.8660j
    - (Orig Idx 1609) q(y) = (2*y_2)/3 + (y_0^2)/3 + (2*y_1^2)/3 + (y_2^2)/3 + (y_0*y_2)/3 + (2*y_1*y_2)/3
      Coefficient: -1.5000 + -0.8660j
    - (Orig Idx 1355) q(y) = (y_2)/3 + (y_1^2)/3 + (2*y_2^2)/3 + (2*y_0*y_1)/3 + (y_0*y_2)/3 + (2*y_1*y_2)/3
      Coefficient: -1.5000 + 0.8660j
    - (Orig Idx 9451) q(y) = (y_0)/3 + (y_1)/3 + (y_2^2)/3 + (2*y_0*y_1)/3 + (2*y_0*y_2)/3 + (2*y_1*y_2)/3
      Coefficient: 1.5000 + -0.8660j

--- Sanity Check (SA Fixed k) ---
Norm of difference (SA fixed k): 3.82e-14
✅ Sanity Check 1 (SA Fixed k) Passed.
✅ Sanity Check 2 (SA Fixed k - Round Trip) Passed.

SECOND TRY (notice the 0 coefficient, this proves the stabrank is less than or equal to 5)

✅ SA Best representation found with fixed k=6:
  Final Cost: 2.6104e-14
  Final Error: 2.61e-14
  Basis functions used:
    - (Orig Idx 364) q(y) = (y_0^2)/3 + (y_1^2)/3 + (y_2^2)/3 + (y_0*y_1)/3 + (y_0*y_2)/3 + (y_1*y_2)/3
      Coefficient: 1.5000 + -0.8660j
    - (Orig Idx 499) q(y) = (y_0^2)/3 + (y_1^2)/3 + (y_2^2)/3 + (2*y_0*y_1)/3
      Coefficient: -0.0000 + -0.0000j
    - (Orig Idx 6579) q(y) = (y_0)/3 + (2*y_0^2)/3
      Coefficient: -1.5000 + 0.8660j
    - (Orig Idx 11431) q(y) = (y_0)/3 + (2*y_1)/3 + (y_0^2)/3 + (y_2^2)/3 + (2*y_0*y_1)/3
      Coefficient: -0.0000 + 1.7321j
    - (Orig Idx 15609) q(y) = (2*y_0)/3 + (y_1)/3 + (y_1^2)/3 + (y_0*y_1)/3 + (2*y_1*y_2)/3
      Coefficient: 1.5000 + -0.8660j
    - (Orig Idx 7213) q(y) = (y_0)/3 + (y_1^2)/3 + (y_2^2)/3 + (2*y_0*y_1)/3 + (2*y_0*y_2)/3
      Coefficient: 1.5000 + -0.8660j

--- Sanity Check (SA Fixed k) ---
Norm of difference (SA fixed k): 2.54e-14
✅ Sanity Check 1 (SA Fixed k) Passed.
✅ Sanity Check 2 (SA Fixed k - Round Trip) Passed.