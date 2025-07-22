# include "read_PonzaInstance.h"

static const regex numRE(R"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)");

auto firstNumber = [](const string &txt) -> double {
    smatch m;
    if (regex_search(txt, m, numRE)) return stod(m.str());
    return 0.0;
};

vector<double> allNumbers(const string &txt) {
    vector<double> v;
    smatch m;
    string s = txt;
    while (regex_search(s, m, numRE)) {
        v.push_back(stod(m.str()));
        s = m.suffix();
    }
    return v;
}

Instance readPonzaInstance(const string &path) {
    ifstream fin(path);
    if (!fin) {
        cerr << "[error] cannot open " << path << "\n";
        exit(1);
    }
    string data((istreambuf_iterator<char>(fin)), {});
    Instance I;
    I.name = path;
    smatch m;

    if (regex_search(data, m, regex(R"(\bc\s*=\s*(\d+))"))) I.c = stoi(m[1]);
    if (regex_search(data, m, regex(R"(\bSL\s*=\s*([0-9eE.+-]+))"))) I.SL = stod(m[1]);
    if (regex_search(data, m, regex(R"(\bSR\s*=\s*([0-9eE.+-]+))"))) I.SR = stod(m[1]);
    if (regex_search(data, m, regex(R"(\bE\s*=\s*([0-9eE.+-]+))"))) I.droneEndurance = stod(m[1]);
    if (regex_search(data, m, regex(R"(\bBKS\s*=\s*([0-9eE.+-]+))"))) I.BKS = stod(m[1]);

    int n = I.c + 2;
    auto parseMatrix = [&](const string &key) {
        size_t pos = data.find(key);
        if (pos == string::npos) return vector<vector<double>>{};
        size_t lb = data.find('[', pos);
        size_t sc = data.find(';', lb);
        string sub = data.substr(lb, sc - lb);
        vector<double> nums = allNumbers(sub);
        vector<vector<double>> M(n, vector<double>(n, 0));
        size_t idx = 0;
        for (int i = 0; i < n && idx < nums.size(); ++i)
            for (int j = 0; j < n; ++j) M[i][j] = nums[idx++];
        return M;
    };
    I.tauTruck = parseMatrix("tauTruck");
    I.tauDrone = parseMatrix("tauDrone");

    if (I.tauTruck.empty() || I.tauDrone.empty()) {
        cerr << "[warn] missing tau matrices in " << path << "\n";
        exit(1);
    }
    return I;
}