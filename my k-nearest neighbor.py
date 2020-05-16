"""
2019
"""
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
li = load_iris()
input_data = []
train_data = []
classes = []
properties_names = []
k_num = 1
"""classes_of_input_data = []
classes_of_train_data = []


def from_sklearn_to_arrays(load_object):
    global input_data, train_data, classes, properties_names
    print("accessing data...")
    for i in range(len(load_object['feature_names'])):
        properties_names.append(load_object['feature_names'][i])
    for i in range(len(load_object['target_names'])):
        classes.append(load_object['target_names'][i])
    x_train, x_test, y_train, y_test = train_test_split(load_object['data'], load_object['target'], random_state=0)
    for i in range(0, int(len(x_test))):
        input_data.append([])
        for v in range(len(x_test[0])):
            input_data[i].append(x_test[i][v])
        classes_of_input_data.append(y_test[i])
    for i in range(int(len(x_train))):
        train_data.append([])
        for v in range(len(x_train[0])):
            train_data[i].append(x_train[i][v])
        train_data[i].append(y_train[i])
        classes_of_train_data.append(y_train[i])
    print("data accessed.")
    return True """


def read_file(file_path):
    split_data = []
    file = open(file_path, "r")
    for line in file:
        split_data.append(line.split(","))
    split_data_int = []
    for i in range(len(split_data)):
        split_data_int.append([])
        for u in range(len(split_data[0])):
            split_data_int[i].append(int(split_data[i][u]))
    return split_data_int


def fill_input_data():
    global k_num, train_data, input_data, classes
    train_data_file_path = input("""
    pls_input the files full path of the file containing train points data in form:
        property1,property2,...,class
        property1,property2,...,class
        ...
    note that the class attribute must be like 0,1,2,3,..,last class num
    file's path: """)
    for i in range(100):
        print("to stop, type [n]o ")
        in_d = input("type class " + str(i) + " name: ")
        if in_d == "n":
            break
        classes.append(in_d)
    test_data_file_path = input("type the path of the file containing the test data in the same form of the past file: ")
    k_num = input("type the number of neighbors k you want to consider: ")
    train_data = read_file(train_data_file_path)
    input_data = read_file(test_data_file_path)


def bubble_sort(arr, position):   # the arr argument takes a multi dimensional array of shape [[2,1],[..]]where 2
    def swap(g, j):
        arr[g], arr[j] = arr[j], arr[g]

    n = len(arr)
    swapped = True
    x = -1
    while swapped:
        swapped = False
        x = x + 1
        for i in range(1, n - x):
            if arr[i - 1][position] > arr[i][position]:
                swap(i - 1, i)
                swapped = True
    return arr


def get_least(data, num, position):
    sorted_data = bubble_sort(data, position)
    return sorted_data[:num]


def compute_distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class PointObject:
    def __init__(self, data):
        self.properties = data    # properties are all characteristics of one point
        # in the same order that is put on the training set
        self.properties_num = len(self.properties)

    def compute_euclidean_distance(self, point):    # computes distance btn the objrct and another point
        total = 0
        for i in range(self.properties_num):
            total = total + (self.properties[i] - point[i]) ** 2
        return math.sqrt(total)

    def sort_wrt_distances(self):
        sorted_distances_classes = []
        distances = []
        for i in range(len(train_data)):
            distances.append([i, self.compute_euclidean_distance(train_data[i])])
        sorted_distances = get_least(distances, k_num, 1)
        for i in range(len(sorted_distances)):
            sorted_distances_classes.append(train_data[sorted_distances[i][0]])
        return sorted_distances_classes  # returns the train_data sorted wrt distances in decreasing order. only k_num
    # points will be returned

    def compute_class_frequency(self, data):
        freqs = []
        for p in range(len(classes)):
            freqs.append([p, 0])
        for i in range(len(data)):
            for k in range(len(classes)):
                if data[i][len(properties_names)] == k:
                    freqs[k][1] = freqs[k][1] + 1
        return freqs

    def pick_class(self):
        train_data_sorted_wrt_distances = self.sort_wrt_distances()
        train_data_class_freqs = self.compute_class_frequency(train_data_sorted_wrt_distances)
        train_data_access_pick_class = get_least(train_data_class_freqs, len(classes), 1)
        class_picked = train_data_access_pick_class[len(classes) - 1][0]
        return class_picked


def score(data):
    test_points = []
    error = 0
    good = 0
    for v in range(len(data)):
        test_points.append(PointObject(data[v]))
    for v in range(len(data)):
        if test_points[v].pick_class() != data[v][len(data[v]) - 1]:
            error = error + 1
        else:
            good = good + 1
    return (good / (good + error)) * 100


def add_classes():
    global input_data
    test_points = []
    for v in range(len(input_data)):
        test_points.append(PointObject(input_data[v]))
        input_data[v].append(test_points[v].pick_class())
    return True


def plot():
    axises_of_input_data, class_of_input_data = [], []
    axises_of_train_data, class_of_train_data = [], []
    for v in range(len(input_data)):
        class_of_input_data.append(input_data[v][len(properties_names)])
    for v in range(len(train_data)):
        class_of_train_data.append(train_data[v][len(properties_names)])
    for v in range(0, len(properties_names)):
        axises_of_input_data.append([])
        for i in range(len(input_data)):
            axises_of_input_data[v].append(input_data[i][v])
    for v in range(0, len(properties_names)):
        axises_of_train_data.append([])
        for i in range(len(train_data)):
            axises_of_train_data[v].append(input_data[i][v])
    fig1, ax1 = plt.subplots(nrows=len(properties_names), ncols=len(properties_names))
    fig1.suptitle("test data graphs")
    for v in range(len(properties_names)):
        for i in range(len(properties_names)):
            if properties_names[v] != properties_names[i]:
                ax1[v][i].scatter(axises_of_input_data[v], axises_of_input_data[i], c=class_of_input_data)
                ax1[v][i].set_title(properties_names[v] + " to " + properties_names[i])
    legend1 = fig1.legend(["1", "2"], loc="lower left", title="Classes")
    fig1.add_artist(legend1)
    plt.subplots_adjust(hspace=.3)
    plt.show()


if __name__ == '__main__':
    print(0)