#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <string>
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
    std::vector<int> xs = {95,2,47,50,52,87,69,79,63,75,44,11,64,49,42,77,59,83,48,60,93,28,73,100,92,26,20,7,73,84};
    std::vector<int> ys = {69,74,32,98,57,48,80,72,5,87,16,71,87,1,64,13,68,23,35,26,66,58,99,82,38,71,52,10,18,12};
    std::vector<Point> points;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        points.push_back(Point(xs[i], ys[i]));
    }
    Tour initialTour(points);
    double initialTemp = 80.0;
    double coolingRate = 0.99999;
    int iterations = 10000000;
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

    plt::figure_size(1200, 780);

    plt::subplot(2, 1, 1);
    plt::plot(initX, initY, "k.-");
    plt::title(std::string("Original tour: ") + std::to_string(initialTour.cost()));
    plt::xlim(0, 100);
    plt::ylim(0, 100);

    plt::subplot(2, 1, 2);
    plt::plot(finalX, finalY, "k.-");
    plt::title(std::string("Final tour: ") + std::to_string(solution.cost()));
    plt::xlim(0, 100);
    plt::ylim(0, 100);

    plt::show();
    plt::save("./solution.png");

    return 0;
}