import math

def sec_to(sec, to):
    min = sec / 60
    hours = min / 60
    days = hours / 24
    mon = days / 30
    year = mon / 12
    tho = year / 1000
    mil = tho / 1000
    if to == "seconds":
        return sec
    if to == "minutes":
        return min
    if to == "hours":
        return hours
    if to == "days":
        return days
    if to == "months":
        return mon
    if to == "years":
        return year
    if to == "thousand years":
        return tho
    if to == "million years":
        return mil


def calc_time(k,m, time):
    routes_per_sec = (pow(2,30)) / (100*(k+m))
    num_routes = math.factorial(k) * pow((m+1),(k)) * m
    scientific_notation = "{:e}".format(num_routes)
    seconds = num_routes / routes_per_sec
    print(f"K={k}, M={m}")
    print(f"Number of routes is {scientific_notation}")
    print(f"time in {time} is {sec_to(seconds, time)}")
    

calc_time(7,2, "seconds")
calc_time(7,3, "minutes")
calc_time(8,3, "hours")
calc_time(8,4, "hours")
calc_time(9,3, "days")
calc_time(10,2, "months")
calc_time(11,3, "years")
calc_time(12,3, "thousand years")
calc_time(12,4, "thousand years")
calc_time(13,4, "million years")


