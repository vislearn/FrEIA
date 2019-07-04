#include<math.h>

void joint_filter(float* l_up, float* a_dw, float* a_up,
                  int n_up, int m_up, int n_dw, int m_dw,
                  double s_x, double s_l) {

    int range_x = ceil(s_x * 3);
    double c_x = 1./(s_x*s_x);
    double c_l = 1./(s_l*s_l);
    double scaling = (double) n_dw / n_up;

    double px_result, px_filt_norm, l, l0, w_x, w_l;
    int i, j, di, dj, di_dw, dj_dw;

    for (i = 0; i < n_up; i++){
        for (j = 0; j < m_up; j++){

            px_result = 0.;
            px_filt_norm = 0.;
            l0 = (double) l_up[i*m_up + j];

            for(di = i-range_x; di < i+range_x; di++){
                if (di < 0 || di >= n_up) continue;
                for(dj = j-range_x; dj < j+range_x; dj++){
                    if (dj < 0 || dj >= m_up) continue;

                    l = (double) l_up[di*m_up + dj];
                    w_x = exp(-0.5 * ((di-i)*(di-i) + (dj-j)*(dj-j)) * c_x);
                    w_l = exp(-0.5 * (l-l0)*(l-l0) * c_l);
                    w_x *= w_l;

                    di_dw = floor(di * scaling);
                    dj_dw = floor(dj * scaling);
                    px_result += w_x * a_dw[di_dw * m_dw + dj_dw];
                    px_filt_norm += w_x;

                }
            }
            a_up[i*m_up + j] = (float) (px_result / px_filt_norm);
        }
    }
}
