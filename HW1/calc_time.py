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


def calc_time(args):
    for k, m, time in args:
        routes_per_sec = (pow(2,30)) / (100*(k+m))
        num_routes = math.factorial(k) * pow((m+1),(k)) * m
        scientific_notation = "{:e}".format(num_routes)
        seconds = num_routes / routes_per_sec
        print(f"K={k}, M={m}")
        print(f"Number of routes is {scientific_notation}")
        print(f"time in {time} is {sec_to(seconds, time)}")
    
args = [(7,2,"seconds"),(7,3,"minutes"),(8,3,"hours"),(8,4,"hours"),(9,3,"days"),(10,3,"months"),(11,3,"years"),(12,3,"thousand years"),(12,4,"thousand years"),(13,4,"million years")]

calc_time(args)


