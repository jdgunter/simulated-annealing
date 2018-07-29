#include <vector>
#include <iostream>
#include <cmath>
#include <random>

struct Point {
    double x;
    double y;
    Point(int _x, int _y) : x(_x), y(_y) {}

    // Euclidean distance between two points
    double distance(const Point& p) {
        double xdist = x - p.x;
        double ydist = y - p.y;
        return std::sqrt(xdist*xdist + ydist*ydist);
    }

    Point& operator=(const Point& p) {
        x = p.x;
        y = p.y;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& stream, const Point& p) {
    stream << "(" << p.x << "," << p.y << ")";
    return stream;
}

class Tour {
private:
    std::vector<Point> tourPoints;
    int computedCost;

    void computeCost() {
        computedCost = 0;
        for (std::size_t i = 0, max = tourPoints.size() - 1; i < max; ++i) {
            computedCost += tourPoints[i].distance(tourPoints[i+1]);
        }
    }

public:
    Tour(std::vector<Point> pts) : tourPoints(pts) { computeCost(); }
    Tour(const Tour& t) : tourPoints(t.points()) { computeCost(); }

    // returns the vector representing the tour
    std::vector<Point> points() const { return tourPoints; }

    // returns the number of points in the tour
    int size() const { return tourPoints.size(); }

    // returns the cost of the tour, i.e. the sum of the euclidean distance between each
    // pair of points in the tour
    int cost() const { return computedCost; }

    // swaps the points at position i and j within the tour and recomputes the cost
    void swap(int i, int j) {
        Point tmp = tourPoints[i];
        tourPoints[i] = tourPoints[j];
        tourPoints[j] = tmp;
        computeCost();
    }
};

class SimulatedAnnealingSolver {
private:
    // immutable members
    const int maxIterations;
    const double coolingRate;

    // mutable members
    Tour state; 
    double currentTemp;

    // random number generator/distributions
    std::mt19937 rng;
    std::uniform_int_distribution<int> neighborDistribution;
    std::uniform_real_distribution<double> zeroToOne;

    // cooling function following a geometric schedule
    // i.e. the temperature at time t is T(t) = T_0 * alpha^t 
    void coolSystem() {
        currentTemp *= coolingRate;
    }

    // function which selects a random state within the state graph
    // adjacent to the current state
    Tour randomNeighbour() {
        int i = neighborDistribution(rng);
        Tour newState = state;
        newState.swap(i,i+1);
        return newState;
    }

    double acceptanceProbability(int energy, int newEnergy, double temp) const {
        if (newEnergy < energy) {
            return 1.0;
        } else {
            return std::exp( (energy - newEnergy) / temp );
        }
    }

public:
    SimulatedAnnealingSolver(Tour initState, double initTemp, double alpha, int iterations) 
    : maxIterations(iterations),
      coolingRate(alpha),
      state(initState), 
      currentTemp(initTemp),  
      rng((std::random_device())()),
      neighborDistribution(0, initState.size()-2),
      zeroToOne(0.0, 1.0) {}

    Tour solve() {
        for (int i = 0; i < maxIterations; ++i) {
            coolSystem();
            Tour neighbor = randomNeighbour();
            if (acceptanceProbability(state.cost(), neighbor.cost(), currentTemp) >= zeroToOne(rng)) {
                state = neighbor;
            }
        }
        return state;
    }
};

int main() {
    std::vector<Point> points {Point(37, 59),
                               Point(40, 58),
                               Point(89, 16),
                               Point(46, 27),
                               Point(88, 25),
                               Point(48, 61),
                               Point(93, 44),
                               Point(45, 50),
                               Point(48, 5),
                               Point(62, 12)};
    Tour initialTour(points);
    double initialTemp = 70.0;
    double alpha = 0.997;
    int iterations = 5000000;
    SimulatedAnnealingSolver sas(initialTour, initialTemp, alpha, iterations);
    Tour solution = sas.solve();

    std::cout << "Initial tour cost: " << initialTour.cost() << "\n";
    std::cout << "Initial tour points: ";
    for (auto p : initialTour.points()) {
        std::cout << p << " ";
    }
    std::cout << "\n";

    std::cout << "Final tour cost: " << solution.cost() << "\n";
    std::cout << "Final tour points: ";
    for (auto p : solution.points()) {
        std::cout << p << " ";
    }
    std::cout << "\n";

    return 0;
}