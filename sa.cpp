#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

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
        computedCost += tourPoints.back().distance(tourPoints.front());
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

    // computes the probability that we transition to the new state, given
    // the current system energy, new system energy, and temperature
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
    std::vector<int> xs = {37, 40, 89, 46, 88, 48, 93, 45, 48, 62};
    std::vector<int> ys = {59, 58, 16, 27, 25, 61, 44, 59, 5,  12};
    std::vector<Point> points;
    for (int i = 0; i < 10; ++i) {
        points.push_back(Point(xs[i], ys[i]));
    }
    Tour initialTour(points);
    double initialTemp = 70.0;
    double coolingRate = 0.9999;
    int iterations = 5000000;
    SimulatedAnnealingSolver sas(initialTour, initialTemp, coolingRate, iterations);
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

    std::vector<int> initX;
    std::vector<int> initY;
    auto initPoints = initialTour.points();
    for (auto p : initPoints) {
        initX.push_back(p.x);
        initY.push_back(p.y);
    }
    initX.push_back(initPoints[0].x);
    initY.push_back(initPoints[0].y);

    std::vector<int> finalX;
    std::vector<int> finalY;
    auto solPoints = solution.points();
    for (auto p : solPoints) {
        finalX.push_back(p.x);
        finalY.push_back(p.y);
    }
    finalX.push_back(solPoints[0].x);
    finalY.push_back(solPoints[0].y);


    plt::subplot(2, 1, 1);
    plt::plot(initX, initY, "k.-");
    plt::title("Original tour");

    plt::subplot(2, 1, 2);
    plt::plot(finalX, finalY, "r.-");
    plt::title("Final tour");

    plt::show();

    return 0;
}