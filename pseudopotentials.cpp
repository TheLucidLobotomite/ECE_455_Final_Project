#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <cmath>

#define PI 3.14159265358979323846

struct UPF {

    std::string element;
    double z_valence;
    bool ultrasoft = false;
    bool paw = false;
    int lmax = -1;
    int mesh = 0;
    int nproj = 0;

    std::vector<double> r, rab;
    std::vector<double> vloc;
    std::vector<std::vector<double>> beta;
    std::vector<int> beta_l;
    std::vector<std::vector<double>> D;
    std::vector<std::vector<double>> Q;
    

};

// trims all the garbage off of the string
std::string trim_string(const std::string& str) {
    size_t start = str.find_first_not_of(" \n\r\t");
    size_t end = str.find_last_not_of(" \n\r\t");
    return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

std::vector<double> read_data_for_tag(std::istream& file, const std::string& tag){
    std::vector<double> data;
    std::string line;
    while (std::getline(file, line)){
        line = trim_string(line);

        if (line.find("</" + tag) != std::string::npos) break;
        if (line.find("<" + tag) != std::string::npos) continue;

        std::stringstream ss(line);
        double val;
        while (ss >> val) data.push_back(val);

    }

    return data;
}

UPF read_upf(const std::string& filename) {
    UPF upf;
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    bool mesh = false, nonlocal = false, aug = false;

    while (std::getline(file, line)){
        line = trim_string(line);

        // attribute extraction
        if (line.find("<PP_HEADER") != std::string::npos) {
            // element
            size_t pos = line.find("element=");
            if (pos != std::string::npos) {
                upf.element = line.substr(pos + 9, 2);
            }
            // z_valence
            if (line.find("z_valence=") != std::string::npos){
                sscanf(line.c_str(), "%*[^z]z_valence=\"%lf\"", &upf.z_valence);
            }
            // ultrasoft
            if (line.find("is_ultrasoft=\"T\"") != std::string::npos || line.find("is_ultrasoft=\"true\"") != std::string::npos){
                upf.ultrasoft = true;
            }
            // PAW
            if (line.find("is_paw=\"T\"") != std::string::npos || line.find("is_paw=\"true\"") != std::string::npos){
                upf.paw = true;
            }
            size_t meshpos = line.find("mesh_size=\"");
            size_t pos2 = line.find("number_of_proj=\"");
            if (meshpos != std::string::npos){
                upf.mesh = std::stoi(line.substr(meshpos + 11, 4));
            }
            if (pos2 != std::string::npos){
                upf.nproj = std::stoi(line.substr(pos2 + 16, 1));
            }
        
        }

        if (line.find("<PP_MESH") != std::string::npos) mesh = true;
        if (line.find("</PP_MESH>") != std::string::npos) mesh = false;

        if (line.find("<PP_NONLOCAL") != std::string::npos) nonlocal = true;
        if (line.find("</PP_NONLOCAL>") != std::string::npos) nonlocal = false;

        if (line.find("<PP_AUGMENTATION") != std::string::npos) aug = true;
        if (line.find("</PP_AUGMENTATION>") != std::string::npos) aug = false;

        // radial grid
        if (mesh) {
            if (line.find("<PP_R>") != std::string::npos){
                upf.r = read_data_for_tag(file, "PP_R");
            }
            if (line.find("<PP_RAB>") != std::string::npos){
                upf.rab = read_data_for_tag(file, "PP_RAB");
            }
        }

        if (line.find("<PP_LOCAL") != std::string::npos){
            upf.vloc = read_data_for_tag(file, "PP_LOCAL");
        }

        // projectors
        if (nonlocal && line.find("<PP_BETA.") != std::string::npos) {
            int n, l = 0;
            size_t pos3 = line.find("<PP_BETA.");
            size_t pos4 = line.find("angular_momentum=");
            int betaindex = std::stoi(line.substr(pos3 + 9, 1)) - 1;
            int angularmomentum = std::stoi(line.substr(pos4 + 18, 1));
            std::vector<double> rawbeta = read_data_for_tag(file, "PP_BETA." + std::to_string(betaindex + 1));
            std::vector<double> beta(upf.mesh);
            for (int i = 0; i < upf.mesh; ++i) {
                beta[i] = (i < rawbeta.size() && rawbeta[i] != 0.0) ? rawbeta[i] / upf.r[i] : 0.0;
            }
            upf.beta.push_back(beta);
            upf.beta_l.push_back(angularmomentum);

        }

        // D_IJ matrix
        if (nonlocal && line.find("<PP_DIJ>") != std::string::npos) {
            std::vector<double> dij = read_data_for_tag(file, "PP_DIJ");
            int n = upf.nproj;
            upf.D.assign(n, std::vector<double>(n, 0.0));
            int k = 0;
            for (int i = 0; i < n; ++i){
                for (int j = 0; j < n; ++j){
                    if (k < (int)dij.size()){
                        upf.D[i][j] = upf.D[j][i] = dij[k];
                        k++;
                    }
                }
            }
        }

        // Q matrix
        if (nonlocal && line.find("<PP_Q>") != std::string::npos) {
            std::vector<double> q = read_data_for_tag(file, "PP_Q");
            int n = upf.nproj;
            upf.Q.assign(n, std::vector<double>(n, 0.0));
            int k = 0;
            for (int i = 0; i < n; ++i){
                for (int j = 0; j < n; ++j){
                    if (k < (int)q.size()){
                        upf.Q[i][j] = upf.Q[j][i] = q[k];
                        k++;
                    }
                }
            }
        }


        // Augmentation Q_ij(r)
        /*if ((upf.ultrasoft || upf.paw) && line.find("<PP_QIJ") != std::string::npos){
            int i,j,l;
            size_t pos = line.find("<PP_QIJL.");
            i = std::stoi(line.substr(pos + 10, 1)) - 1;
            j = std::stoi(line.substr(pos + 12, 1)) - 1;
            l = std::stoi(line.substr(pos + 14, 1));
            std::vector<double> rawq = read_data_for_tag(file,"PP_QIJ");
        }*/
    }

    return upf;

}

double Vnl_GGp(UPF& upf, double G, double GPrime) {
    double q = abs(G - GPrime);           // q = |G - G'| in bohr⁻¹
    double form_factor = 0.0;
    for (int i = 0; i < upf.r.size(); ++i) {
        if (upf.r[i] < 1e-12) continue;
        double dr = (i==0) ? upf.r[1]-upf.r[0] : upf.r[i]-upf.r[i-1];
        double x = q * upf.r[i];
        double j0 = (x < 1e-8) ? 1.0 : sin(x)/x;        // spherical Bessel j₀(x)
        form_factor += upf.beta[0][i] * j0 * upf.r[i]*upf.r[i] * upf.rab[i];  // ∫ β(r) j₀(qr) r² dr
    }
    form_factor *= 4.0 * PI;
    return upf.D[0][0] * form_factor;   // this is ⟨G|Vnl|G′⟩
}
/*
int main(){
    UPF test;
    std::string filename = "C:\\Users\\Charlie\\Documents\\ECE 455\\Test\\Test.UPF";
    test = read_upf(filename);

    return 1;
}
*/


