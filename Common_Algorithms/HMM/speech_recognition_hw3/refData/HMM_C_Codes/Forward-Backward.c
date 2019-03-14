const int N = 3, M = 3, T = 15;
double pi[N], a[N][N], b[N][M]; // HMM
double alpha[T][N];                // 可以簡化成alpha[2][N]
double beta[T][N];                // 可以簡化成beta[2][N]
//γ : gamma, ξ : epsilon
//pi : pi,    ψ : psi
double forward(int *o, int T)
{
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < N; ++j)
            if (t == 0)
                alpha[t][j] = pi[j] * b[j][o[t]];
            else
            {
                double p = 0;
                for (int i = 0; i < N; ++i)
                    p +=alpha[t - 1][i] * a[i][j];
                alpha[t][j] = p * b[j][o[t]];
            }

    double p = 0;
    for (int i = 0; i < N; ++i)
        p += alpha[T - 1][i];
    return p;
}

double backward(int *o, int T)
{
    for (int t = T - 1; t >= 0; --t)
        for (int i = 0; i < N; ++i)
            if (t == T - 1)
                beta[t][i] = 1.0;
            else
            {
                double p = 0;
                for (int j = 0; j < N; ++j)
                    p += a[i][j] * b[j][o[t + 1]] * beta[t + 1][j];
                beta[t][i] = p;
            }

    double p = 0;
    for (int j = 0; j < N; ++j)
        p += pi[j] * b[j][o[0]] * beta[0][j];
    return p;
}