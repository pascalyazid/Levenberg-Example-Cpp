#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <unsupported/Eigen/NonLinearOptimization>

using namespace std;
using namespace Eigen;

double calc_distance(const VectorXd &anchor, VectorXd &tag_device) {
  return sqrt((pow((anchor(0) - tag_device(0)), 2) +
               pow((anchor(1) - tag_device(1)), 2) +
               pow((anchor(2) - tag_device(2)), 2)));
}

double residual(const Eigen::VectorXd &p, const Eigen::Vector3d &a,
                double measured_distance) {
  return std::sqrt(((p - a).array().square()).sum()) - measured_distance;
}

template <typename Scalar> struct Residuals {
  using ScalarType = Scalar;
  int m_inputs, m_values;

  Residuals(int inputs, int values, const Eigen::MatrixXd &anchors,
            const Eigen::VectorXd &distances)
      : m_inputs(inputs), m_values(values), anchors_(anchors),
        distances_(distances) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

  int operator()(const Eigen::VectorXd &p, VectorXd &fvec) const {
    Eigen::VectorXd residuals(anchors_.rows());

    for (int i = 0; i < anchors_.rows(); ++i) {
      fvec(i) = residual(p, anchors_.row(i), distances_[i]);
    }

    return 0;
  }

  int df(const Eigen::VectorXd &p, Eigen::MatrixXd &fjac) const {
    Eigen::Matrix<double, 7, 1> epsilon;

    double epsilon0 = 0.0001;

    epsilon(0) = epsilon0 * 1; // Pg_x
    epsilon(1) = epsilon0 * 1; // Pg_y
    epsilon(2) = epsilon0 * 1; // Pg_z
    epsilon(3) = epsilon0 * 1; // v_x
    epsilon(4) = epsilon0 * 1; // v_y
    epsilon(5) = epsilon0 * 1; // v_z
    epsilon(6) = epsilon0 * 1; // t_0
    for (int i = 0; i < p.size(); i++) {
      Eigen::VectorXd pPlus(p);
      pPlus(i) += epsilon(i);
      Eigen::VectorXd pMinus(p);
      pMinus(i) -= epsilon(i);

      Eigen::VectorXd fvecPlus(values());
      operator()(pPlus, fvecPlus);

      Eigen::VectorXd fvecMinus(values());
      operator()(pMinus, fvecMinus);

      Eigen::VectorXd fvecDiff(values());
      fvecDiff = fvecPlus - fvecMinus;
      // fix for epsilon(0-1)
      if (epsilon(i) != 0)
        fvecDiff /= 2.0f * epsilon(i);

      fjac.block(0, i, values(), 1) = fvecDiff;
    }

    // std::cout << fjac << std::endl;
    return 0;
  }

private:
  Eigen::MatrixXd anchors_;
  Eigen::VectorXd distances_;
};

Eigen::Vector3d multilaterate(const Eigen::Vector3d *a, const double *r,
                              const size_t count, const Vector3d &guess) {

  MatrixXd anchors((int)count, 3);
  for (int i = 0; i < (int)count; i++) {
    anchors.row(i) = a[i];
  }

  VectorXd distances((int) count);
  for (int i = 0; i < (int)count; i++) {
    distances(i) = r[i];
  }

  VectorXd initial_guess(3);
  initial_guess = guess;


  std::cout
      << "\n-----------------------------------------------------------\n";
  std::cout << "\nDistances:\n" << distances << "\n";

  std::cout << "\nInitial guess:\n" << initial_guess << "\n";

  Residuals<double> functor(initial_guess.size(), anchors.rows(), anchors,
                            distances);

  LevenbergMarquardt<Residuals<double>> lm(functor);
  lm.minimize(initial_guess);
  std::cout << "\nResult:\n"
            << fixed << initial_guess(0) << ", " << fixed << initial_guess(1)
            << ", " << fixed << initial_guess(2) << "\n";
  std::cout
      << "\n-----------------------------------------------------------\n";

  return initial_guess;
}

int main() {
  //TODO input vectors from command line input or from file
  Vector3d anchor01(0.0, 0.0, 0.0);
  Vector3d anchor02(2.0, 2.0, 0.0);
  Vector3d anchor03(2.0, 0.0, 2.0);
  Vector3d anchor04(0.0, 2.0, 2.0);

  const Vector3d a[4]{anchor01, anchor02, anchor03, anchor04};
  const double r[4] = {1.73, 1.73, 1.73, 1.73};

  const size_t count = 4;

  Vector3d guess(3);
  guess << 0.0, 0.0, 0.0;

  multilaterate(a, r, count, guess);
}
