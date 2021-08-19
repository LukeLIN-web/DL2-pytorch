import multiprocessing


def f1(x):
    return x * x


def func(msg):
    print(multiprocessing.current_process().name + '-' + msg)


def compare(debug=False):
    pool = multiprocessing.Pool(processes=40)
    print(pool.map(f1, range(10)))

    # for i in range(10):  # one example takes about 10s
    #     # thread_list[0].append(pool.apply_async(drf, args=(traces[i],), error_callback=print_error))
    #     # thread_list[1].append(pool.apply_async(srtf, args=(traces[i],), error_callback=print_error))
    #     # thread_list[2].append(pool.apply_async(fifo, args=(traces[i],), error_callback=print_error))
    #     msg = "hello %d" % i
    #     pool.apply_async(func, (msg,))
    # pool.close()
    # pool.join()


def main():
    print("en")
    compare(False)
    print("we")


if __name__ == '__main__':
    main()
