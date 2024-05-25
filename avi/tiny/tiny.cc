#include <TinyAD/Scalar.hh>

using ADouble = TinyAD::Double<1, false>;

ADouble circle(ADouble radius, Eigen::Vector2d pt) {
    return pt.norm() - radius;
}

ADouble inside(ADouble distance) {
    ADouble e = exp(distance * 10);
    return 1.0 - (e / (e + 1.0));
}

Eigen::Vector2d grid_point(int x, int y, double range, int steps) {
    double step = range / steps;
    double offset = range / 2.0;
    Eigen::Vector2d v = Eigen::Vector2d(step * x - offset, step * y - offset);
    return v;
}

ADouble find_area(int range, int steps, ADouble r) {
    ADouble avg = 0.0;
    double i = 0.0;
    for (int x = 0; x < steps; x++)
    {
        for (int y = 0; y < steps; y++) {
            Eigen::Vector2d pt = grid_point(x, y, range, steps);
            ADouble v = inside(circle(r, pt));
            i += 1.0;
            avg += (v - avg) / i;
        }
    }

    ADouble area = avg * (range * range);
    return area;
}

int main() {
    ADouble r = ADouble::make_active({1.0})[0];
    for (int i = 0; i < 1000; i++) {
        ADouble area = find_area(20.0, 100, r);
        ADouble diff = area - 100.0;
        ADouble cost = diff * diff;
        double new_r = r.val - (cost.grad[0] * 0.0001);
        r = ADouble::make_active({new_r})[0];
    }
    printf("%f\n", r.val);
}