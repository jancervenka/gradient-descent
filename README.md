# Gradient Descent

Example of gradient descent computing simple linear regression. Written in C.

## Build

```bash
gcc grad_desc.c -o grad_desc.out -lm
```

## Run

```bash
./grad_desc.out
```

### Example Output

```
Computing regression coefficients using gradient descent.
Dataset size n=2000
Gradient descent finished after 100000 steps with loss=0.085
Estimated coefficients: a=4.002, b=2.483
Elapsed CPU time: 1.429 seconds
```
