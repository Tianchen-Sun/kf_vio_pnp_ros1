#pragma once
// =============================================================================
// GateMap  –  loads gate world-frame poses from a CSV (scene.csv).
// GatePoseArrayDecoder  –  decodes a one-hot geometry_msgs/PoseArray.
// =============================================================================

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <cmath>

namespace kf_vio_pnp {

// ---------------------------------------------------------------------------
// GateMap
// ---------------------------------------------------------------------------
class GateMap
{
public:
    struct GatePose {
        double cx, cy, cz;
        double yaw_rad;
    };

    explicit GateMap(const std::string& csv_path)
    {
        loadCsv(csv_path);
        if (gates_.empty())
            throw std::runtime_error("GateMap: no gate rows found in " + csv_path);
    }

    /// Retrieve the pose of a gate by its zero-padded two-digit string id, e.g. "07".
    const GatePose& get(const std::string& gate_id_str) const
    {
        auto it = gates_.find(normalizeId(gate_id_str));
        if (it == gates_.end())
            throw std::runtime_error("GateMap: gate_id '" + gate_id_str + "' not found");
        return it->second;
    }

    /// Overload accepting integer gate id.
    const GatePose& get(int gate_id) const { return get(intToId(gate_id)); }

    static std::string normalizeId(const std::string& raw)
    {
        std::string s = raw;
        // Strip "gate_" prefix if present
        if (s.size() >= 5 && s.substr(0, 5) == "gate_")
            s = s.substr(5);
        // Zero-pad to 2 digits
        if (s.size() == 1) s = "0" + s;
        return s;
    }

    static std::string intToId(int n)
    {
        return n < 10 ? ("0" + std::to_string(n)) : std::to_string(n);
    }

private:
    std::unordered_map<std::string, GatePose> gates_;

    void loadCsv(const std::string& path)
    {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("GateMap: cannot open " + path);

        std::string line;
        // First line: header – find column indices
        if (!std::getline(f, line))
            throw std::runtime_error("GateMap: empty file " + path);

        std::vector<std::string> headers = splitCsv(line);
        int col_type = findCol(headers, "row_type");
        int col_name = findCol(headers, "name");
        int col_cx   = findCol(headers, "cx");
        int col_cy   = findCol(headers, "cy");
        int col_cz   = findCol(headers, "cz");
        int col_yaw  = findCol(headers, "yaw_deg");

        while (std::getline(f, line)) {
            auto cols = splitCsv(line);
            if (static_cast<int>(cols.size()) <= std::max({col_type, col_name,
                                                            col_cx, col_cy, col_cz, col_yaw}))
                continue;
            if (cols[col_type] != "gate") continue;

            GatePose p;
            p.cx      = std::stod(cols[col_cx]);
            p.cy      = std::stod(cols[col_cy]);
            p.cz      = std::stod(cols[col_cz]);
            p.yaw_rad = std::stod(cols[col_yaw]) * M_PI / 180.0;

            gates_[normalizeId(cols[col_name])] = p;
        }
    }

    static std::vector<std::string> splitCsv(const std::string& line)
    {
        std::vector<std::string> result;
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            // Trim whitespace
            size_t s = tok.find_first_not_of(" \t\r\n");
            size_t e = tok.find_last_not_of(" \t\r\n");
            result.push_back(s == std::string::npos ? "" : tok.substr(s, e - s + 1));
        }
        return result;
    }

    static int findCol(const std::vector<std::string>& hdr, const std::string& name)
    {
        for (int i = 0; i < static_cast<int>(hdr.size()); ++i)
            if (hdr[i] == name) return i;
        throw std::runtime_error("GateMap: column '" + name + "' not found in CSV header");
    }
};

// ---------------------------------------------------------------------------
// GateDetection  –  result of decoding a PoseArray
// ---------------------------------------------------------------------------
struct GateDetection {
    int              gate_id;    // logical gate index (0-based)
    Eigen::Vector3d  position;   // [x, y, z] in quadrotor (camera) frame
    bool             is_front;   // true = drone approaching (even array index)
};

// ---------------------------------------------------------------------------
// GatePoseArrayDecoder
//
// PoseArray index layout (one-hot):
//   even index 2k   → gate k, front (approaching)
//   odd  index 2k+1 → gate k, back  (departing)
// Only the single non-zero element is considered valid.
// ---------------------------------------------------------------------------
class GatePoseArrayDecoder
{
public:
    /// Decode from a vector of geometry_msgs/Pose objects.
    /// Returns true and fills `det` if a valid detection is found.
    template<typename PoseVec>
    static bool decode(const PoseVec& poses, GateDetection& det)
    {
        for (std::size_t i = 0; i < poses.size(); ++i) {
            const double x = poses[i].position.x;
            const double y = poses[i].position.y;
            const double z = poses[i].position.z;
            if (x == 0.0 && y == 0.0 && z == 0.0) continue;

            det.gate_id  = static_cast<int>(i) / 2;
            det.is_front = (i % 2 == 0);
            det.position = Eigen::Vector3d(x, y, z);
            return true;
        }
        return false;
    }
};

} // namespace kf_vio_pnp
