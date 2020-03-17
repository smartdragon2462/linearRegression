import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

# /////////////////////////////////////////////
def preprocessing():
    le = LabelEncoder()
    min_max = MinMaxScaler()

    school_data = pd.read_csv('2018 dropout-graduation statistics VA.csv')
    school_data["FEDERAL_RACE_CODE"] = school_data["FEDERAL_RACE_CODE"].replace({np.nan:3})
    # CT_school_data["FEDERAL_RACE_CODE"].fillna(3, inplace=True)
    # tt = CT_school_data["FEDERAL_RACE_CODE"].map({np.nan:3})
    school_data["GENDER"] = school_data["GENDER"].replace({np.nan: 1,'M':1,'F':2})
    school_data["DISABILITY_FLAG"] = school_data["DISABILITY_FLAG"].replace({np.nan: 2, 'Y': 1, 'N': 2})
    school_data["LEP_FLAG"] = school_data["LEP_FLAG"].replace({np.nan : 2, 'Y': 1, 'N': 2})
    school_data["DISADVANTAGED_FLAG"] = school_data["DISADVANTAGED_FLAG"].replace({np.nan: 2, 'Y': 1, 'N': 2})

    le.fit(school_data["SCH_NAME"].values)
    school_data["SCH_NAME"] = le.transform(school_data["SCH_NAME"].values)

    m_independence_colmun = ['SCH_NAME','FEDERAL_RACE_CODE','GENDER','DISABILITY_FLAG','LEP_FLAG','DISADVANTAGED_FLAG','COHORT_CNT','DIPLOMA_RATE']
    m_dependence_colmun = ['DROPOUT_RATE']

    X = school_data[m_independence_colmun]
    Y = school_data[m_dependence_colmun]

    X1 = min_max.fit_transform(X)

    # Split X and Y in test and evaluate set
    X_train, X_test, Y_train, Y_test = train_test_split(X1, Y, test_size=0.2, random_state=42)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    print(school_data.head(100))
    print(school_data.columns)
    print(school_data[['DROPOUT_RATE']].head())

    # all_data = pd.DataFrame(X1,columns=m_independence_colmun)
    m_independence_colmun.append('DROPOUT_RATE')
    return X_train, X_test, Y_train, Y_test, school_data[m_independence_colmun]

# /////////////////////////////////////////////
def main():

    X_train, X_test, Y_train, Y_test, all_data = preprocessing()

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 5))
    # f, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    m_attribue = all_data.columns
    for i in range(1,len(m_attribue)):
        plt.subplot(3,3,i)
        sns.distplot(all_data[m_attribue[i]], kde=False)
        plt.xlabel(m_attribue[i])
        plt.grid()
    plt.show()

    corr = all_data.select_dtypes(include=['float64', 'int64']).iloc[:, 1:].corr()
    # fig = plt.figure()
    sns.set(font_scale=1)
    sns.heatmap(corr, vmax=1, square=True)
    plt.title('Correlation Graph for each indepency variables')
    plt.show()



    corr_list = corr['DROPOUT_RATE'].sort_values(axis=0, ascending=False).iloc[1:]
    plt.figure(figsize=(18, 8))
    for i in range(6):
        ii = '23' + str(i + 1)
        plt.subplot(ii)
        feature = corr_list.index.values[i]
        plt.scatter(all_data[feature], all_data['DROPOUT_RATE'], facecolors='none', edgecolors='r', s=2)
        sns.regplot(x=feature, y='DROPOUT_RATE', data=all_data, scatter=False, color='Blue')
        ax = plt.gca()
        # ax.set_ylim([0, 800000])
    plt.show()

    # plt.figure(figsize=(12, 6))
    # sns.boxplot(x='SCH_NAME', y='DROPOUT_RATE', data=all_data)
    # xt = plt.xticks(rotation=45)

    # CUSTOMIZABLE: Collect/Prepare data
    # datapoint_size = 1000
    n = len(X_train)
    n_dim = X_train.shape[1]
    batch_size = 1000
    m_step = np.int16(n/batch_size)
    epoch = 2000
    learn_rate = 0.01

    # Model linear regression y = Wx + b
    x = tf.placeholder(tf.float32, [None, n_dim], name="x")
    W = tf.Variable(tf.zeros([n_dim, 1]), name="W")
    b = tf.Variable(tf.zeros([1]), name="b")
    with tf.name_scope("Wx_b") as scope:
        product = tf.matmul(x, W)
        y = product + b

    # Add summary ops to collect data
    W_hist = tf.summary.histogram("weights", W)
    b_hist = tf.summary.histogram("biases", b)
    y_hist = tf.summary.histogram("y", y)

    y_ = tf.placeholder(tf.float32, [None, 1])

    # Cost function sum((y_-y)**2)
    with tf.name_scope("cost") as scope:
        cost = tf.reduce_mean(tf.square(y_ - y))
        cost_sum = tf.summary.scalar("cost", cost)

    # Training using Gradient Descent to minimize cost
    with tf.name_scope("train") as scope:
        train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

    sess = tf.Session()
    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter(log_file, sess.graph_def)
    sess.run(tf.global_variables_initializer())

    all_feed = {x: X_train, y_: Y_train}
    m_mes = []
    for i in range(epoch):
        for n in range(m_step-1):
            batch_xs = X_train[batch_size*n:batch_size*(n+1)]
            batch_ys = Y_train[batch_size*n:batch_size*(n+1)]

            xs = np.array(batch_xs)
            ys = np.array(batch_ys)

            feed = {x: xs, y_: ys}
            sess.run(train_step, feed_dict=feed)

        if i % 50 == 0:
            # result = sess.run(merged, feed_dict=all_feed)
            # writer.add_summary(result, i)

            pred_y = sess.run(y, feed_dict={x: X_test})
            plt.clf()
            plt.scatter(Y_test, pred_y,c='b', marker='.',label='origin vs predict')
            plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2,label='Linear curve')
            plt.xlabel('Measured')
            plt.ylabel('Predicted')
            plt.legend()
            plt.grid()
            plt.pause(0.1)
            print("%d epoch: \t cost: %0.5f" % (i,sess.run(cost, feed_dict=all_feed)))
            # print("cost: %f" % sess.run(cost, feed_dict=all_feed))

        m_mes.append(sess.run(cost, feed_dict=all_feed))
        # pred_y = sess.run(y, feed_dict={x: X_test})
        # mse = tf.reduce_mean(tf.square(pred_y - Y_test))
        # m_mes.append(sess.run(mse))

    plt.close()

    print("W: %s" % sess.run(W))
    print("b: %f" % sess.run(b))
    print("cost: %f" % sess.run(cost, feed_dict=all_feed))

    pred_y = sess.run(y, feed_dict={x: X_test})
    # cost = tf.reduce_mean(tf.square(pred_y - Y_test))
    # print("Cost: %.4f" % sess.run(cost))
    # print(pred_y)

    sess.close()

    plt.plot(m_mes,'r',label='cost')
    plt.xlabel('Epoch'); plt.ylabel('cost'); plt.title('Cost vs Epoch')
    plt.grid()
    plt.legend()
    plt.show()
    plt.close()

    plt.plot(Y_test, 'r', label='Test')
    plt.plot(pred_y, 'g', label='predict')
    plt.xlabel('number');
    plt.ylabel('DROPOUT_RATE');
    plt.title('DROPOUT_RATE vs number')
    plt.grid()
    plt.legend()
    plt.show()



    # #
    #
    # # n = len(X_train)
    # #
    # # n_dim = X_train.shape[1]
    # # learning_rate = 0.01
    # # training_epochs = 1000
    # # cost_history = np.empty(shape=[1], dtype=float)
    # #
    # # X = tf.placeholder(tf.float32, [None, n_dim])
    # # Y = tf.placeholder(tf.float32, [None, 1])
    # # W = tf.Variable(tf.ones([n_dim, 1]))
    # #
    # # init = tf.initialize_all_variables()
    # # y_ = tf.matmul(X, W)
    # # cost = tf.reduce_mean(tf.square(y_ - Y))
    # # training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # #
    # # sess = tf.Session()
    # # sess.run(init)
    # #
    # # for epoch in range(training_epochs):
    # #     sess.run(training_step, feed_dict={X: X_train, Y: Y_train})
    # #     cost_history = np.append(cost_history, sess.run(cost, feed_dict={X: X_train, Y: Y_train}))
    # #
    # #
    # # plt.plot(range(len(cost_history)), cost_history)
    # # plt.axis([0, training_epochs, 0, np.max(cost_history)])
    # # plt.show()
    # #
    # # pred_y = sess.run(y_, feed_dict={X: X_test})
    # # mse = tf.reduce_mean(tf.square(pred_y - Y_test))
    # # print("MSE: %.4f" % sess.run(mse))
    # #
    # # sess.close()
    # #
    # # fig, ax = plt.subplots()
    # # ax.scatter(Y_test, pred_y)
    # # ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=3)
    # # ax.set_xlabel('Measured')
    # # ax.set_ylabel('Predicted')
    # # plt.show()
    # #
    # #
    # # # X = tf.placeholder("float")
    # # # Y = tf.placeholder("float")
    # # #
    # # # W = tf.Variable(np.random.randn(), name="W")
    # # # b = tf.Variable(np.random.randn(), name="b")
    # # #
    # # # learning_rate = 0.01
    # # # training_epochs = 1000
    # # #
    # # # y_pred = tf.add(tf.multiply(X, W), b)
    # # #
    # # # cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n)
    # # #
    # # # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # # #
    # # #
    # # # # Starting the Tensorflow Session
    # # # with tf.Session() as sess:
    # # #     # Initializing the Variables
    # # #     sess.run(tf.global_variables_initializer())
    # # #
    # # #     # Iterating through all the epochs
    # # #     for epoch in range(training_epochs):
    # # #
    # # #         # Feeding each data point into the optimizer using Feed Dictionary
    # # #         for (_x, _y) in zip(X_train, X_test):
    # # #             sess.run(optimizer, feed_dict={X: _x, Y: _y})
    # # #
    # # #             # Displaying the result after every 50 epochs
    # # #         if (epoch + 1) % 50 == 0:
    # # #             # Calculating the cost at every epoch
    # # #             c = sess.run(cost, feed_dict={X: X_train, Y: X_test})
    # # #             print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b))
    # # #
    # # #             # Storing necessary values to be used outside the Session
    # # #     training_cost = sess.run(cost, feed_dict={X: X_train, Y: X_test})
    # # #     weight = sess.run(W)
    # # #     bias = sess.run(b)
# /////////////////////////////////////////////
if __name__ == "__main__":
    main()