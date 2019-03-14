//γ : gamma, ξ : epsilon
//π : pi,    ψ : psi,     δ : delta
/***********************************/
Forward - Backward Algorithm
/***********************************/

const int N = 3,
M = 3, T = 15;
double π[N], a[N][N], b[N][M]; // HMM
double α[T][N];                // 可以簡化成α[2][N]
double β[T][N];                // 可以簡化成β[2][N]

double forward(int *o, int T)
{
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < N; ++j)
            if (t == 0)
                α[t][j] = π[j] * b[j][o[t]];
            else
            {
                double p = 0;
                for (int i = 0; i < N; ++i)
                    p += α[t - 1][i] * a[i][j];
                α[t][j] = p * b[j][o[t]];
            }

    double p = 0;
    for (int i = 0; i < N; ++i)
        p += α[T - 1][i];
    return p;
}

double backward(int *o, int T)
{
    for (int t = T - 1; t >= 0; --t)
        for (int i = 0; i < N; ++i)
            if (t == T - 1)
                β[t][i] = 1.0;
            else
            {
                double p = 0;
                for (int j = 0; j < N; ++j)
                    p += a[i][j] * b[j][o[t + 1]] * β[t + 1][j];
                β[t][i] = p;
            }

    double p = 0;
    for (int j = 0; j < N; ++j)
        p += π[j] * b[j][o[0]] * β[0][j];
    return p;
}

/***********************************/
          Viterbi Algorithm
/***********************************/

const int N = 3, M = 3, T = 15;
double π[N], a[N][N], b[N][M];  // HMM
double δ[T][N]; // 可以簡化成δ[2][N]
int ψ[T][N];
 
double decode(int* o, int T, int* q)
{
    for (int t=0; t<T; ++t)
        for (int j=0; j<N; ++j)
            if (t == 0)
                δ[t][j] = π[j] * b[j][o[t]];
            else
            {
                double p = -1e9;
                for (int i=0; i<N; ++i)
                {
                    double w = δ[t-1][i] * a[i][j];
                    if (w > p) p = w, ψ[t][j] = i;
                }
                δ[t][j] = p * b[j][o[t]];
            }
 
    double p = -1e9;
    for (int j=0; j<N; ++j)
        if (δ[T-1][j] > p)
            p = δ[T-1][j], q[T-1] = j;
 
    for (int t=T-1; t>0; --t)
        q[t-1] = ψ[t][q[t]];
 
    return p;
}

/***********************************/
Baum - Welch Algorithm
    /***********************************/
    const int N = 3,
              M = 3, T = 15;
double π[N], a[N][N], b[N][M]; // HMM
double α[T][N], β[T][N];       // evaluation problem
double δ[T][N];
int ψ[T][N];                // decoding problem
double γ[T][N], ξ[T][N][N]; // learning problem

void learn(int *o, int T)
{
    forward(o, T);
    backward(o, T);

    for (int t = 0; t < T; ++t)
    {
        double p = 0;
        for (int i = 0; i < N; ++i)
            p += α[t][i] * β[t][i];
        assert(p != 0);

        for (int i = 0; i < N; ++i)
            γ[t][i] = α[t][i] * β[t][i] / p;
    }

    for (int t = 0; t < T - 1; ++t)
    {
        double p = 0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                p += α[t][i] * a[i][j] * b[j][o[t + 1]] * β[t + 1][j];
        assert(p != 0);

        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                ξ[t][i][j] = α[t][i] * a[i][j] * b[j][o[t + 1]] * β[t + 1][j] / p;
    }

    // 更新Π
    for (int i = 0; i < N; ++i)
        π[i] = γ[0][i];

    // 更新A
    for (int i = 0; i < N; ++i)
    {
        double p2 = 0;
        for (int t = 0; t < T - 1; ++t)
            p2 += γ[t][i];
        assert(p2 != 0);

        for (int j = 0; j < N; ++j)
        {
            double p1 = 0;
            for (int t = 0; t < T - 1; ++t)
                p1 += ξ[t][i][j];
            a[i][j] = p1 / p2;
        }
    }

    // 更新B
    for (int i = 0; i < N; ++i)
    {
        double p[M] = {0}, p2 = 0;
        for (int t = 0; t < T; ++t)
        {
            p[o[t]] += γ[t][i];
            p2 += γ[t][i];
        }
        assert(p2 != 0);

        for (int k = 0; k < M; ++k)
            b[i][k] = p[k] / p2;
    }
}