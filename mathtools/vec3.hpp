#ifndef VEC3_H
#define VEC3_H

inline static double safePhi(double phi) {
    if (phi >= 2 * MY_PI)
        phi -= 2 * MY_PI;
    else if (phi < 0)
        phi += 2 * MY_PI;
    return phi;
}

class Vec3 {
public:
    double theta, phi;
    double r[3];
    double cos_theta, sin_theta, cos_phi, sin_phi;
    Vec3() {};
    Vec3(double t, double p) { setAngles(t, p); };

    void setAngles(double t, double p) {
        theta = t; phi = safePhi(p);
        _calcFromAngle();
    }

    double cos_angle(const Vec3 &other) const {
        return sin_theta * other.sin_theta
               + cos_theta * other.cos_theta * cos(other.phi - phi);
    }

    // void rotatePhi(double shift_phi) {
    //     phi += shift_phi;
    //     if (phi >= 2 * MY_PI)
    //         phi -= 2 * MY_PI;
    //     else if (phi < 0)
    //         phi += 2 * MY_PI;

    // }
    void rotate(const std::array<double, 9> &rot_mat) {
        double new_r[3];
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0,
                    rot_mat.data(), 3, r, 1, 0, new_r, 1);
        std::copy_n(new_r, 3, r);
        _calcFromUnitVec();
    }

    static std::array<double, 9> getRotationMatrix(const Vec3 &u, bool transpose=true) {
        std::array<double, 9> rot_mat {
            u.cos_theta * u.cos_phi, -u.sin_phi, u.sin_theta * u.cos_phi,
            u.cos_theta * u.sin_phi, u.cos_phi, u.sin_theta * u.sin_phi,
            -u.sin_theta, 0.0, u.cos_theta
        };

        if (transpose) {
            for (int i = 0; i < 2; i++)
                for (int j = i + 1; j < 3; j++)
                    std::swap(rot_mat[j + i * 3], rot_mat[i + j * 3]);
        }

        return rot_mat;
    }

private:
    void _calcFromAngle() {
        cos_theta = cos(theta);
        sin_theta = sin(theta);
        cos_phi = cos(phi);
        sin_phi = sin(phi);

        r[0] = sin_theta * cos_phi;
        r[1] = sin_theta * sin_phi;
        r[2] = cos_theta;
    }

    void _calcFromUnitVec() {
        theta = acos(r[2]);
        phi = safePhi(atan2(r[1], r[0]) + MY_PI);

        cos_theta = cos(theta);
        sin_theta = sin(theta);
        cos_phi = cos(phi);
        sin_phi = sin(phi);
    }
};

#endif
