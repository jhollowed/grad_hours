import pdb
import sys
import time
import pendulum
import numpy as np
from toggl import api
from pendulum import date
from pendulum import datetime
import matplotlib.pyplot as plt

window_arg = int(sys.argv[1])
print('\n ---- window={}'.format(window_arg))

stop = pendulum.now()
years_back = 4
entries = []
update_entries=False
tmp_fnames = ['duration.npy', 'project.npy', 'day.npy']

try:
    duration = np.load(tmp_fnames[0])
    project = np.load(tmp_fnames[1])
    day = np.load(tmp_fnames[2])
    day_obj = [pendulum.parse(d) for d in day]
    print('Read entry info from file')
    if day_obj[0] != pendulum.now(): update_entries=True

except FileNotFoundError:
    print('Reading entry info via Toggl API')

    duration = np.array([])
    project = np.array([])
    day = np.array([])
    
    for i in range(years_back):
        start = date(year=stop.year-1, month=stop.month, day=stop.month)
        print('fetching for {} - {}'.format(start, stop))
        entries = list(api.TimeEntry.objects.all_from_reports(start, stop))
        stop = start
        
        offset = len(day)
        duration = np.hstack([duration, np.empty(len(entries))])
        project  = np.hstack([project, np.empty(len(entries), dtype=str)])
        day      = np.hstack([day, np.empty(len(entries), dtype=str)])
    
        for j in range(len(entries)):
            print('--- entry {}/{}'.format(j+1, len(entries)), end='\r', flush=True)
            duration[offset+j]     = entries[j].duration/60
            day[offset+j]          = str(entries[j].start)
            try: 
                project[offset+j]  = entries[j].project.name
            except attributeerror: 
                project[offset+j]  = 'none'
            time.sleep(0.01)  # to avoid toggl api throttling

        print('--- writing to file')
        np.save(tmp_fnames[0], duration)
        np.save(tmp_fnames[1], project)
        np.save(tmp_fnames[2], day)
        update_entries=False

while(update_entries):
    print('--- updating entry data')
    start = day_obj[0]
    stop= pendulum.now()
    entries = list(api.TimeEntry.objects.all_from_reports(start, stop))
    # this seems to still return entries from before the start time, i.e. it only cares
    # about the calendar day, not the time. Manually remove those which are duplicates
    entries[:] = [e for e in entries if e in day_obj]

    print('found {} new entries'.format(len(entries)))
    if(len(entries)== 0): 
        break
 
    offset = len(day)
    duration = np.hstack([duration, np.empty(len(entries))])
    project  = np.hstack([project, np.empty(len(entries), dtype=str)])
    day      = np.hstack([day, np.empty(len(entries), dtype=str)])

    for j in range(len(entries)):
        print('--- entry {}/{}'.format(j+1, len(entries)), end='\r', flush=True)
        duration[offset+j]     = entries[j].duration/60
        day[offset+j]          = str(entries[j].start)
        try: 
            project[offset+j]  = entries[j].project.name
        except attributeerror: 
            project[offset+j]  = 'none'
        time.sleep(0.01)  # to avoid toggl api throttling

    day_obj = np.hstack([day_obj, [pendulum.parse(d) for d in day]])
    print('--- writing to file')
    pdb.set_trace()
    np.save(tmp_fnames[0], duration)
    np.save(tmp_fnames[1], project)
    np.save(tmp_fnames[2], day)
    update_entries=False
    

# organize projects into just these 3 "keepers"
# be smart about some categories, randomly assign hours from others
# (there are very few of these)
keep_proj = ['Research', 'Coursework', 'Teaching']
all_proj = np.unique(project)
np.random.seed(0)
for i in range(len(project)):
    if(project[i]== 'FV3'): project[i] = 'Research'
    elif(project[i]== 'CLDERA'): project[i] = 'Research'
    elif(project[i]== 'GSI'): project[i] = 'Teaching'
    elif(project[i] not in keep_proj): project[i] = keep_proj[np.random.randint(0,3)]

# group each entry by date to get total daily hours per project
dates = np.array([d.split('T')[0] for d in day])
period = pendulum.period(day_obj[-1], day_obj[0])
all_dates = [dt.format('YYYY-MM-DD') for dt in  period.range('days')]
all_dates_obj = [dt for dt in period.range('days')]

hours_per_day = np.zeros((len(all_dates), len(keep_proj)))
for i in range(len(all_dates)):
    if (all_dates[i] not in dates):
        hours_per_day[i,:] = 0
    else:
        these_dates = dates[dates == all_dates[i]]
        these_proj  = project[dates == all_dates[i]]
        these_dur   = duration[dates == all_dates[i]]
        for k in range(len(keep_proj)):
            hours_per_day[i,k] += sum(these_dur[these_proj == keep_proj[k]])/60

# get tot cumulative hours
tot_cumul_hours = np.cumsum(hours_per_day, axis=0)

# take week-long rolling sum
hours_per_day_tot = np.sum(hours_per_day, axis=1)
hours_per_day_rolling = np.zeros(hours_per_day.shape)

window = window_arg
#window = 7
weight = 7/window
for i in range(len(keep_proj)):
    hours_per_day_rolling[:,i] = np.convolve(hours_per_day[:,i], np.ones(window), 'same') * weight

tot_window = window
tot_weight = 7/tot_window
hours_per_day_rolling_tot = np.convolve(hours_per_day_tot, np.ones(tot_window), 'same') * tot_weight

print('cumulative hours: {:.2f}'.format(np.sum(hours_per_day)))
print('cumulative smoothed hours: {:.2f}'.format(np.sum(hours_per_day_rolling)/7))
norm = np.sum(hours_per_day) / (np.sum(hours_per_day_rolling)/7)
hours_per_day_rolling = hours_per_day_rolling * norm
print('normalized cumulative smoothed hours: {:.2f}'.format(np.sum(hours_per_day_rolling)/7))
#tot_cumul_hours = np.cumsum(hours_per_day_rolling/7, axis=0)


# plot
fig = plt.figure(figsize=(10, 3))
ax = fig.add_subplot(111)
ax2 = ax.twinx()

colors = ['orange', 'c', 'm']
for i in range(len(keep_proj)):
    p = ax.plot(all_dates_obj, hours_per_day_rolling[:,i], color=colors[i], 
                label='{} rolling weekly mean'.format(keep_proj[i]))[0]
    ax2.plot(all_dates_obj, tot_cumul_hours[:,i], color=p.get_color(), 
             ls='-', alpha=0.5, lw=1.0,
             label='{} cumulative'.format(keep_proj[i]))
#ax.plot(all_dates_obj, hours_per_day_rolling_tot, label='total')

unique_years = np.unique([d.year for d in all_dates_obj])
for year in unique_years:
    mask = np.array([d.year == year for d in all_dates_obj])
    p = ax.plot(np.array(all_dates_obj)[mask], 
                np.ones(len(np.array(all_dates)[mask]))*np.mean(hours_per_day_rolling_tot[mask]), 
                'k--', lw=1.1, zorder=0)[0] 
    if(year == 2022): p.set_label('cal. yearly means')

ax.grid(alpha=0.2)
ax.legend(loc='upper left')
ax.set_xlim([all_dates_obj[0], all_dates_obj[-1]])
ax.set_ylim([0, 62])
ax2.set_ylim([0, 1100])
plt.tight_layout()
plt.savefig('./figs/{:03d}.png'.format(window), dpi=150)
