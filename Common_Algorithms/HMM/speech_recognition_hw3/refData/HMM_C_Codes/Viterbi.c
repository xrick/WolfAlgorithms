const int N = 3, M = 3, T = 15;
double π[N], a[N][N], b[N][M]; // HMM
double δ[T][N];                // 可以簡化成δ[2][N]
int ψ[T][N];

double Viterbi_Decode(int *o, int T, int *q)
{
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < N; ++j)
            if (t == 0)
                δ[t][j] = π[j] * b[j][o[t]];
            else
            {
                double p = -1e9;
                for (int i = 0; i < N; ++i)
                {
                    double w = δ[t - 1][i] * a[i][j];
                    if (w > p)
                        p = w, ψ[t][j] = i;
                }
                δ[t][j] = p * b[j][o[t]];
            }

    double p = -1e9;
    for (int j = 0; j < N; ++j)
        if (δ[T - 1][j] > p)
            p = δ[T - 1][j], q[T - 1] = j;

    for (int t = T - 1; t > 0; --t)
        q[t - 1] = ψ[t][q[t]];

    return p;
}