#pragma once
// =============================================================================
// Online 2nd-order Butterworth low-pass filter
//
// Implemented as a Direct-Form II biquad IIR filter.
// Filter state is seeded on the first sample to eliminate the DC startup
// transient (equivalent to scipy's sosfilt_zi initialisation).
//
// Coefficients are computed analytically via the bilinear transform of the
// 2nd-order Butterworth analogue prototype.
// =============================================================================

#include <cmath>
#include <vector>
#include <array>
#include <stdexcept>
#include <string>

namespace kf_vio_pnp {

// ---------------------------------------------------------------------------
// Biquad coefficient set (normalised: a0 = 1)
// Difference equation (Direct Form II):
//   w[n]  =  x[n]  − a1·w[n−1] − a2·w[n−2]
//   y[n]  = b0·w[n] + b1·w[n−1] + b2·w[n−2]
// ---------------------------------------------------------------------------
struct BiquadCoeffs {
    double b0, b1, b2;   // numerator
    double a1, a2;        // denominator (feedback, a0 = 1)
};

// Compute 2nd-order Butterworth LPF coefficients via bilinear transform.
// cutoff_hz : desired −3 dB cutoff frequency  (Hz)
// sample_hz : sample rate of the discrete system (Hz)
inline BiquadCoeffs butterworth2ndOrder(double cutoff_hz, double sample_hz)
{
    // Pre-warp analogue cutoff (bilinear transform)
    const double wc   = std::tan(M_PI * cutoff_hz / sample_hz);
    const double wc2  = wc * wc;
    const double norm = wc2 + wc * std::sqrt(2.0) + 1.0;

    BiquadCoeffs c;
    c.b0 =  wc2 / norm;
    c.b1 = 2.0 * wc2 / norm;
    c.b2 =  wc2 / norm;
    c.a1 =  2.0 * (wc2 - 1.0) / norm;
    c.a2 = (wc2 - wc * std::sqrt(2.0) + 1.0) / norm;
    return c;
}

// ---------------------------------------------------------------------------
// OnlineButterworthFilter
// Filters a vector of `channels` scalars, one sample at a time.
// ---------------------------------------------------------------------------
class OnlineButterworthFilter
{
public:
    OnlineButterworthFilter() = default;

    /// @param cutoff_hz  Low-pass −3 dB cutoff (Hz)
    /// @param sample_hz  Nominal sample rate of input data (Hz)
    /// @param channels   Number of parallel scalar channels (e.g. 3 for xyz)
    OnlineButterworthFilter(double cutoff_hz, double sample_hz, int channels = 3)
        : channels_(channels)
        , initialized_(false)
    {
        const double nyq = sample_hz / 2.0;
        if (cutoff_hz <= 0.0 || cutoff_hz >= nyq)
            throw std::invalid_argument(
                "OnlineButterworthFilter: cutoff_hz (" + std::to_string(cutoff_hz)
                + ") must be strictly inside (0, Nyquist=" + std::to_string(nyq) + ")");

        c_ = butterworth2ndOrder(cutoff_hz, sample_hz);

        // Each channel keeps two delay-line values  [w[n-1], w[n-2]]
        state_.assign(channels, {0.0, 0.0});
    }

    /// Filter one sample.  x must have `channels` elements.
    /// Returns filtered vector of the same size.
    std::vector<double> filter(const std::vector<double>& x)
    {
        if (!initialized_) {
            // Seed state so that a DC input x[0] produces output x[0] immediately.
            // Steady-state of the biquad for DC input x_ss:
            //   w_ss = x_ss / (1 + a1 + a2)
            // DC gain of a Butterworth LPF = 1, so y_ss = x_ss (verified algebraically).
            for (int ch = 0; ch < channels_; ++ch) {
                const double denom = 1.0 + c_.a1 + c_.a2;
                const double w_ss  = (std::abs(denom) > 1e-12)
                                        ? x[ch] / denom
                                        : x[ch];
                state_[ch][0] = w_ss;   // w[n-1]
                state_[ch][1] = w_ss;   // w[n-2]
            }
            initialized_ = true;
        }

        std::vector<double> y(channels_);
        for (int ch = 0; ch < channels_; ++ch) {
            const double w_n  = x[ch]
                                - c_.a1 * state_[ch][0]
                                - c_.a2 * state_[ch][1];
            y[ch] = c_.b0 * w_n
                  + c_.b1 * state_[ch][0]
                  + c_.b2 * state_[ch][1];

            state_[ch][1] = state_[ch][0];  // shift delay line
            state_[ch][0] = w_n;
        }
        return y;
    }

    // Convenience overload for Eigen Vector3d
    Eigen::Vector3d filterVec3(const Eigen::Vector3d& v)
    {
        std::vector<double> in = {v(0), v(1), v(2)};
        auto out = filter(in);
        return Eigen::Vector3d(out[0], out[1], out[2]);
    }

    bool isInitialized() const { return initialized_; }

private:
    int                                    channels_    = 3;
    bool                                   initialized_ = false;
    BiquadCoeffs                           c_{};
    std::vector<std::array<double, 2>>     state_;   // per-channel [w[n-1], w[n-2]]
};

} // namespace kf_vio_pnp
