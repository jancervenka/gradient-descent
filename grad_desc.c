/*
 * Example of computing linear regression using gradient descent
 * Jan Cervenka
 * jan.cervenka@yahoo.com
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define LEARNING_RATE 0.001
#define STEPS 100000
#define X_UB 20
#define NOISE_UB 1
#define DATA_SIZE 2000
#define TRUE_A 4
#define TRUE_B 2

struct ab {
    double a, b;
};

/*
 * get_random_number
 * --------------------
 * Produces a random number between 0 and upper_bound.
 * 
 * upper_bound: maximum value of the produced number
 * 
 * returns: random double
 */
double get_random_number(int upper_bound) {

    return ((double) rand() / (double) RAND_MAX) * upper_bound;
}

/*
 * get_random_data
 * --------------------
 * Produces a random matrix containing one feature (x)
 * and regression target (y).
 * 
 * n: size of the dataset
 * true_coefs: coefficients for computing x from y
 * x_ub: maximum value of x
 * noise_ub: noise amplitude
 * 
 * returns: pointer to the dataset
 */
double** get_random_data(int n, struct ab true_coefs, int x_ub, int noise_ub) {

    double *x, *y;
    double **data;

    x = (double*) malloc(n * sizeof(double));
    y = (double*) malloc(n * sizeof(double));
    data = (double**) malloc(2 * sizeof(double*));

    for (int i = 0; i < n; i++) {
        *(x + i) = get_random_number(x_ub);
        *(y + i) = (true_coefs.a * (*(x + i)) + true_coefs.b) + get_random_number(noise_ub);
    }

    *(data + 0) = x;
    *(data + 1) = y;

    return data;
}

/*
 * get_loss
 * --------------------
 * Computes value of loss function for given regression coefficients.
 * 
 * data: training dataset
 * n: size of the dataset
 * current_coefs: regression coefficients
 * 
 * returns: loss value
 */
double get_loss(double **data, int n, struct ab current_coefs) {
    double x, y_true, y_pred, square_sum = 0;

    for (int i = 0; i < n; i++) {
        x = *(*(data + 0) + i);
        y_true = *(*(data + 1) + i);
        y_pred = x * current_coefs.a + current_coefs.b;
        
        square_sum += pow((y_true - y_pred), 2);
    }
    
    return square_sum / n;
}

/*
 * get_loss_gradient
 * --------------------
 * Computes gradient in given location (defined by current_coefs).
 * 
 * data: training dataset
 * n: size of the dataset
 * current_coefs: regression coefficients
 * 
 * returns: gradient
 */
struct ab get_loss_gradient(double **data, int n, struct ab current_coefs) {
    double x, y_true, a_grad = 0, b_grad = 0;
    struct ab grad;

    for (int i = 0; i < n; i++) {
        x = *(*(data + 0) + i);
        y_true = *(*(data + 1) + i);

        a_grad += -x * (y_true - (current_coefs.a * x + current_coefs.b));
        b_grad += -(y_true - (current_coefs.a * x + current_coefs.b));
    }

    grad.a = (a_grad * 2) / n;
    grad.b = (b_grad * 2) / n;

    return grad;
}

/*
 * do_step
 * --------------------
 * Moves one step in the gradient direction.
 * 
 * data: training dataset
 * n: size of the dataset
 * current_coefs: regression coefficients
 * learning_rate: step size
 * 
 * returns: new coefficients
 */
struct ab do_step(double **data, int n, struct ab current_coefs, double learning_rate) {
    struct ab grad, new_coefs;

    grad = get_loss_gradient(data, n, current_coefs);
    new_coefs.a = current_coefs.a - learning_rate * grad.a;
    new_coefs.b = current_coefs.b - learning_rate * grad.b;

    return new_coefs;
}

/*
 * main
 * --------------------
 * Runs the program.
 */
void main() {

    int i, n = DATA_SIZE;
    double current_loss, cpu_time_used;
    clock_t t_start, t_end;
    struct ab true_coefs, current_coefs;

    true_coefs = (struct ab) {.a = TRUE_A, .b = TRUE_B};
    current_coefs = (struct ab) {.a = 1, .b = 0};

    double **data = get_random_data(n, true_coefs, X_UB, NOISE_UB);

    printf("Computing regression coefficients using gradient descent.\n");
    printf("Dataset size n=%d\n", n);

    t_start = clock();
    for (i = 0; i < STEPS; i++) {
        current_coefs = do_step(data, n, current_coefs, LEARNING_RATE);
    }
    t_end = clock();
    cpu_time_used = ((double) (t_end - t_start)) / CLOCKS_PER_SEC;

    current_loss = get_loss(data, n, current_coefs);
    printf("Gradient descent finished after %d steps with loss=%.3f\n", i, current_loss);
    printf("Estimated coefficients: a=%.3f, b=%.3f\n", current_coefs.a, current_coefs.b);
    printf("Elapsed CPU time: %.3f seconds \n", cpu_time_used);

    free(*(data + 0));
    free(*(data + 1));
    free(data);
}
