// From: https://stackoverflow.com/questions/49206780/column-wise-initialization-and-calculation-of-standard-deviation-in-eigen-librar
double std_dev(Eigen::ArrayXd vec) {
    return std::sqrt(
        (vec - vec.mean()).square().sum() / (vec.size() - 1)
    );
}


static double f(const Eigen::VectorXd& w, const Eigen::VectorXd& solution) {
    return -(w - solution).array().square().sum();
}


Eigen::VectorXd randn(int dim) {
    static std::default_random_engine       generator;
    static std::normal_distribution<double> distribution(0.0, 1.0);

    Eigen::VectorXd w(dim);
    w << distribution(generator),
         distribution(generator),
         distribution(generator);
    return w;
}


Eigen::MatrixXd randn(int a, int b) {
    static std::default_random_engine       generator;
    static std::normal_distribution<double> distribution(0.0, 1.0);

    Eigen::MatrixXd w(a, b);
    for (int i=0; i < a; i++) {
        for (int j=0; j < b; j++) {
            w(i, j) = distribution(generator);
        }
    }

    return w;
}


static void es_test() {
    Eigen::Matrix<double, 3, 1> solution;
    Eigen::VectorXd w = randn(3);

    const unsigned int npop { 50 };
    const float sigma { 0.1f };
    const float alpha { 0.001f };

    solution = { 0.5, 0.1, -0.3 };

    for (int i=0; i < 300; i++) {
        if (i % 20 == 0) {
            printf("iter: %d w: %.2f, %.2f, %.2f solution: %.2f, %.2f, %.2f reward: %.2f\n",
                   i,
                   w[0], w[1], w[2],
                   solution[0], solution[1], solution[2],
                   f(w, solution)
            );
        }

        Eigen::MatrixXd N = randn(npop, 3);
        Eigen::VectorXd R(npop);

        for (int j=0; j < npop; j++) {
            Eigen::VectorXd row = N.row(j);
            Eigen::VectorXd w_try = w + row * sigma;
            R(j) = f(w_try, solution);
        }

        Eigen::VectorXd A = (R.array() - R.mean()) / std_dev(R);
        const float f = alpha / (npop * sigma);
        Eigen::VectorXd na = N.transpose() * A;

        // Update weights
        w = w + f * na;
    }
}
