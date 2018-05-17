#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

//=============================================================================
//  @brief: UKF()
//          Constructor for Unscented Kalman Filter (UKF) class
//          Initializes Kalman filter
//
//  @params:  none 
//  @return:  none 
//=============================================================================
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // Initial state vector
  x_ = VectorXd( n_x_ );

  // Initial covariance matrix
  P_ = MatrixXd( n_x_, n_x_ );

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  // Augmented state dimension
  n_aug_ = 7;

  n_sigma_points_ = 2*n_aug_ + 1;

  // Spreading parameter for sigma points
  lambda_ = 3 - n_aug_;

  // NIS
  NIS_radar_ = 0.;
  NIS_lidar_ = 0.;

  // Previous measurement time
  time_us_ = 0;

  // Set weights
  weights_ = VectorXd( n_sigma_points_ );
  weights_(0) = lambda_/( lambda_ + n_aug_ );
  for( int i=1; i<n_sigma_points_; i++ ) {
    weights_(i) = (double) ( 0.5 / ( n_aug_ + lambda_ ) );
  }

  // Initialize state vector
  x_ = VectorXd( n_x_ );
  x_.fill(1.);

  // Initialze covariance matrix
  P_ = MatrixXd( n_x_, n_x_ );
  P_.fill(0.);
  P_ = MatrixXd::Identity( n_x_, n_x_ );

  is_initialized_ = false;
}

//=============================================================================
//  @brief: ~UKF()
//          Desctructor for Unscented Kalman Filter (UKF) class
//          Frees resources for the Kalman filter class
//
//  @params:  none 
//  @return:  none 
//=============================================================================
UKF::~UKF() {}

//=============================================================================
//  @brief: ProcessMeasurement()
//          First pass: Initialize Radar and Lidar Vectors
//          Subsequent passes: Process latest Radar and Lidar measurements
//
//  @params:  meas_package
//  @return:  void 
//=============================================================================
void UKF::ProcessMeasurement( MeasurementPackage meas_package ) {

  //  Initialization
  if( !is_initialized_ ) {
    InitializeStateVector( meas_package );
    return;
  }

  //  Prediction
  const double delta_t = ( meas_package.timestamp_ - time_us_ ) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  Prediction( delta_t );

  //  Update
  if( use_radar_ && ( meas_package.sensor_type_ == MeasurementPackage::RADAR ) ) {
    UpdateRadar( meas_package );
  }
  else if( use_laser_ && ( meas_package.sensor_type_ == MeasurementPackage::LASER ) ) {
    UpdateLidar( meas_package );
  }
  else {
    cout << "Error:" << __func__ << ": Unknown sensor_type:" << meas_package.sensor_type_ << endl;
    return;
  }

  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
}

//=============================================================================
//  @brief: Prediction()
//          Estimate the object's location. 
//          Predict sigma points, the state, and the state covariance matrix.
//
//  @params:  double delta_t 
//  @return:  void 
//=============================================================================
void UKF::Prediction( const double delta_t ) {

  CreateAugmentedSigmaPoints();

  PredictSigmaPoints( delta_t );

  PredictMeanAndCovariance();
}

//=============================================================================
//  @brief: NormalizeAngle()
//          Normalize angle measurement to be between +/- 2xPI radians
//
//  @params: double angle          
//  @return: double angle 
//=============================================================================
double UKF::NormaliseAngle( double& angle ) {

  while( angle > M_PI ) 
    angle -= 2.*M_PI;

  while( angle < -M_PI ) 
    angle += 2.*M_PI;

  return angle;
}

//=============================================================================
//  @brief: InitializeStateVector()
//          Initialize the state vector using first measurement for
//          Radar and Lidar sensors
//
//  @params:  MeasurementPackage meas_package
//  @return:  void 
//=============================================================================
void UKF::InitializeStateVector( MeasurementPackage meas_package ) {

  time_us_ = meas_package.timestamp_;

  if( meas_package.sensor_type_ == MeasurementPackage::RADAR ) {
    x_ = InitializeRadarVector( meas_package.raw_measurements_ );
  } 
  else if( meas_package.sensor_type_ == MeasurementPackage::LASER ) {
    x_ = InitializeLidarVector( meas_package.raw_measurements_ );
  }
  else {
    cout << "Error:" << __func__ << ": Unknown sensor_type:" << meas_package.sensor_type_ << endl;
    return;
  }

  is_initialized_ = true; 
}

//=============================================================================
//  @brief: PolarToCartersion()
//          Converts 1x3 Polar to 1x5 Cartersion vector
//          and returning initialized 1x5 vector.
//  @params: 1x3 Polar Vector 
//  @return: 1x5 Cartesian Vector 
//=============================================================================
VectorXd UKF::PolarToCartesian( const VectorXd &v_in ) {

  VectorXd v_out( n_x_ );

  const auto rho = v_in(0);
  const auto phi = v_in(1);
  const auto rho_dot = v_in(2);

  const auto px = rho * cos( phi );
  const auto py = rho * sin( phi );
  const auto vx = rho_dot * cos( phi );
  const auto vy = rho_dot * sin( phi );

  v_out << px, py, sqrt( vx*vx + vy*vy ), 0, 0;

  return v_out;
}

//=============================================================================
//  @brief: InitializeRadarVector()
//            Initializes a Radar Vector by converting 1x3 Polar vector to a 
//            1x4 Cartersion vector and returning initialized 1x4 vector
//  @params:  1x3 Polar Vector 
//  @return:  1x4 Cartesian Vector 
//=============================================================================
VectorXd UKF::InitializeRadarVector( const VectorXd &v_in ) {

  VectorXd v_out( n_x_ );
  v_out.fill( 1.0 );

  // Check input state parameters for a potential division by zero
  if( v_in(0) == 0 && v_in(1) == 0 ) {
    cout << "Error:" << __func__ << ": Division by zero detected" << endl;
    return v_out;
  }

  v_out = PolarToCartesian( v_in );

  return v_out;
}

//=============================================================================
//  @brief: InitializeLidarVector()
//          Initializes a Lidar Vector which is already in Cartersion form 
//  @params: 1x4 Catesian Vector 
//  @return: 1x4 Initialized Lidar Vector 
//=============================================================================
VectorXd UKF::InitializeLidarVector( const VectorXd &v_in ) {

  VectorXd v_out( n_x_ );

  v_out << v_in( 0 ),v_in( 1 ), 0, 0, 0;

  return v_out;
}

//=============================================================================
//  @brief: CreateAugmentedSigmaPoints()
//          Create the augumented sigma points: 
//          Sampling P_ with sigma points and augment the state vectors with the 
//          mean of the process noise values (have zero mean).
//  @params: none
//  @return: void
//=============================================================================
void UKF::CreateAugmentedSigmaPoints() {

  // Augmented sigma points
  Xsig_aug_ = MatrixXd( n_aug_, n_sigma_points_ );

  VectorXd x_aug = VectorXd( n_aug_ );

  // Create augmented state covariance
  MatrixXd P_aug = MatrixXd( n_aug_, n_aug_ );

  // Create augmented mean state
  x_aug.head( n_x_ ) = x_;
  x_aug( n_x_ ) = 0;
  x_aug( n_x_ + 1 ) = 0;

  // Create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner( 5, 5 ) = P_;
  P_aug( 5, 5 ) = std_a_ * std_a_;
  P_aug( 6, 6 ) = std_yawdd_ * std_yawdd_;

  // Create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // Create augmented sigma points
  Xsig_aug_.col(0)  = x_aug;

  for( int i = 0; i< n_aug_; i++ ) {
    Xsig_aug_.col( i+1 )          = x_aug + sqrt( lambda_ + n_aug_ ) * L.col(i);
    Xsig_aug_.col( i+1 + n_aug_ ) = x_aug - sqrt( lambda_ + n_aug_ ) * L.col(i);
  }
}

//=============================================================================
//  @brief: PredictSigmaPoints()
//          Run all the sigma points through the nonlinear process model to get
//          the sigma points at k+1|k.
//
//  @params: double delta_t
//  @return: void
//=============================================================================
void UKF::PredictSigmaPoints( const double delta_t ) {

  // Predicted sigma points
  Xsig_pred_ = MatrixXd( n_x_, n_sigma_points_ );

  // Predict sigma points
  for( int i = 0; i< n_sigma_points_; i++ ) {
    // Extract values for better readability
    const auto p_x = Xsig_aug_( 0,i );
    const auto p_y = Xsig_aug_( 1,i );
    const auto v = Xsig_aug_( 2,i );
    const auto yaw = Xsig_aug_( 3,i );
    const auto yawd = Xsig_aug_( 4,i );
    const auto nu_a = Xsig_aug_( 5,i );
    const auto nu_yawdd = Xsig_aug_( 6,i );

    // Predicted state values
    double px_p, py_p;

    // Avoid division by zero
    if( fabs( yawd ) > 0.001 ) {
      px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw) );
      py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
      px_p = p_x + v*delta_t*cos( yaw );
      py_p = p_y + v*delta_t*sin( yaw );
    }

    auto v_p = v;
    auto yaw_p = yaw + yawd*delta_t;
    auto yawd_p = yawd;

    // Add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos( yaw );
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin( yaw );
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // Avoid zero division
    if( fabs( px_p ) < 0.0001 && fabs( py_p ) < 0.0001 ) 
      px_p = 0.0001;

    // Write predicted sigma point into right column
    Xsig_pred_( 0,i ) = px_p;
    Xsig_pred_( 1,i ) = py_p;
    Xsig_pred_( 2,i ) = v_p;
    Xsig_pred_( 3,i ) = yaw_p;
    Xsig_pred_( 4,i ) = yawd_p;
  }
}

//=============================================================================
//  @brief: PredictMeanAndCovariance()
//          Update the state (x_) and covariance (P_) using the predicted 
//          sigma points.
//
//  @params: none          
//  @return: void
//=============================================================================
void UKF::PredictMeanAndCovariance() {

  // Predicted state mean
  x_.fill(0.0);

  for( int i = 0; i < n_sigma_points_; i++ ) {
    x_ = x_+ weights_(i) * Xsig_pred_.col(i);
  }

  // Predicted state covariance matrix
  P_.fill(0.0);

  for( int i = 0; i < n_sigma_points_; i++ ) {
    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = NormaliseAngle( x_diff(3) );
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

//=============================================================================
//  @brief: UpdateLidar()
//          Updates the state and the state covariance matrix using a laser
//          measurement. x_ and P_ are updated from predictions at current time 
//          based on previous time measurment k+1|k to current time based on 
//          current time measurement k+1|k+1
//
//  @params:  MeasurementPackage meas_package 
//  @return:  void 
//=============================================================================
void UKF::UpdateLidar( MeasurementPackage meas_package ) {

  int n_z = 2;

  VectorXd z = meas_package.raw_measurements_;

  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd( n_z, n_sigma_points_ );

  // Transform sigma points into measurement space
  for( int i = 0; i < n_sigma_points_; i++ ) {
    // Extract values for better readibility
    const auto p_x = Xsig_pred_( 0,i );
    const auto p_y = Xsig_pred_( 1,i );

    // Measurement model
    Zsig( 0,i ) = p_x;
    Zsig( 1,i ) = p_y;
  }

  // Mean predicted measurement z_pred
  VectorXd z_pred = VectorXd( n_z );
  z_pred.fill(0.0);

  for( int i=0; i < n_sigma_points_; i++ ) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd( n_z,n_z );
  S.fill(0.0);
  
  for( int i = 0; i < n_sigma_points_; i++ ) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  MatrixXd R = MatrixXd( n_z, n_z );
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;
  S = S + R;

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd( n_x_, n_z );

  // Calculate cross correlation matrix
  Tc.fill(0.0);

  for( int i = 0; i < n_sigma_points_; i++ ) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = NormaliseAngle( x_diff(3) );

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z_diff = z - z_pred;

  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Calculate NIS
  NIS_lidar_ = ( z - z_pred ).transpose() * S.inverse() * ( z - z_pred );
  cout << "NIS Lidar: " << NIS_lidar_ << endl;

}

//=============================================================================
//  @brief: UpdateRadar()
//          Update the state (x_) and covariance (P_) using the predicted 
//          sigma points
//          Updates the state (x_) and the state covariance matrix (P_) using a 
//          radar measurement from predictions at current time based on 
//          previous time measurment k+1|k to current time based on current time 
//          measurement k+1|k+1
//
//  @params:  MeasurementPackage meas_package 
//  @return: void
//=============================================================================
void UKF::UpdateRadar( MeasurementPackage meas_package ) {

  const int n_z = 3;

  VectorXd z = meas_package.raw_measurements_;

  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd( n_z, n_sigma_points_ );

  // Transform sigma points into measurement space
  for( int i = 0; i < n_sigma_points_; i++ ) {
    // Extract values for better readibility
    const auto p_x = Xsig_pred_( 0,i );
    const auto p_y = Xsig_pred_( 1,i );
    const auto v   = Xsig_pred_( 2,i );
    const auto yaw = Xsig_pred_( 3,i );

    const auto v1 = cos( yaw ) * v;
    const auto v2 = sin( yaw ) * v;

    // Measurement model
    Zsig( 0,i ) = sqrt( p_x*p_x + p_y*p_y );
    Zsig( 1,i ) = atan2( p_y,p_x );
    Zsig( 2,i ) = ( p_x*v1 + p_y*v2 ) / Zsig(0,i);
  }

  // Mean predicted measurement z_pred
  VectorXd z_pred = VectorXd( n_z );
  z_pred.fill(0.0);

  for( int i=0; i < n_sigma_points_; i++ ) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd( n_z, n_z );
  S.fill(0.0);

  for( int i = 0; i < n_sigma_points_; i++ ) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = NormaliseAngle( z_diff(1) );
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  MatrixXd R = MatrixXd( n_z, n_z );
  R <<  std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0,std_radrd_*std_radrd_;
  S = S + R;

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd( n_x_, n_z );

  // Calculate cross correlation matrix
  Tc.fill(0.0);

  for( int i = 0; i < n_sigma_points_; i++ ) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = NormaliseAngle( z_diff(1) );

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = NormaliseAngle( x_diff(3) );

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z_diff = z - z_pred;
  z_diff(1) = NormaliseAngle( z_diff(1) );

  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Calculate NIS
  NIS_radar_ = ( z - z_pred ).transpose() * S.inverse() * ( z - z_pred );
  cout << "NIS Radar: " << NIS_radar_ << endl;
}
