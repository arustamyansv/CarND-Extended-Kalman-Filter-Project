#include "kalman_filter.h"
using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::UpdateRegular(const VectorXd &z) {
  VectorXd y = z - H_ * x_;

  this->Update(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Convert cortesian to polar coordinates
  float ro = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
  float theta = atan2(x_(1), x_(0));
  float ro_dot;

  // Check if distance is too small to avoid divide by zero
  if (fabs(ro) < 0.0001) {
    ro_dot = 0;
  } else {
    ro_dot = (x_(0)*x_(2) + x_(1)*x_(3))/ro;
  }

  VectorXd z_pred(3);
  z_pred << ro,
            theta,
            ro_dot;

  VectorXd y = z - z_pred;

  // Make theta between -PI and PI
  while (y(1) > M_PI) {
    y(1) -= 2 * M_PI;
  }
  while (y(1) < -M_PI) {
    y(1) += 2 * M_PI;
  }

  // Run common update process
  this->Update(y);
}

void KalmanFilter::Update(const VectorXd &y) {
  // Calculate Calman gain
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // Apply measurement
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}