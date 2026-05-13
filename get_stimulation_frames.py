import parse_nidq as pni
import pandas as pd

def get_stimulation_times_df(base_folder):
    bs = pni.get_binary_signals(base_folder, overwrite=False)
    bs_md = bs.metadata
    # get all rows of bs_md where column 'event' contains 'chrimson_on' or 'chr2_on'
    on_events = bs_md[bs_md['event'].str.contains('chrimson_on|chr2_on')]
    off_events = bs_md[bs_md['event'].str.contains('chrimson_off|chr2_off')]
    #make a dataframe with columns 'event', 'time', 'type' where type is 'on' or 'off'
    on_event_times = []
    for index, row in on_events.iterrows():
        ontimes = bs[index].t
        for t in ontimes:
            on_event_times.append({'event': row['event'], 'time': t, 'type': 'on'})
    off_event_times = []
    for index, row in off_events.iterrows():
        offtimes = bs[index].t
        for t in offtimes:
            off_event_times.append({'event': row['event'], 'time': t, 'type': 'off'})
    on_event_times_df = pd.DataFrame(on_event_times)
    off_event_times_df = pd.DataFrame(off_event_times)
    event_times_df = pd.concat([on_event_times_df, off_event_times_df], ignore_index=True)
    #order by time
    event_times_df = event_times_df.sort_values(by='time')


def get_stimulation_times(base_folder, nidaq_map=None, overwrite=False):
    # snippet for testing
    #from pathlib import Path
    #base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251209")
    #bs = pni.get_binary_signals(base_folder, overwrite=False)

    if nidaq_map is not None:
        dinmap = nidaq_map.get('dinmap', None)
        ainmap = nidaq_map.get('ainmap', None)
        optoidx = nidaq_map.get('optoidx', None)
    else:
        dinmap=None
        ainmap=None
        optoidx=None

    bs = pni.get_binary_signals(base_folder, dinmap=dinmap, ainmap=ainmap, optoidx=optoidx, overwrite=overwrite, plot=True)

    bs_md = bs.metadata
    # get all rows of bs_md where column 'event' contains 'chrimson_on' or 'chr2_on'
    on_events = bs_md[bs_md['event'].str.contains('chrimson_on|chr2_on|solenoid_on')]
    off_events = bs_md[bs_md['event'].str.contains('chrimson_off|chr2_off')]
    # for each row in on_events, make a list of time times
    on_event_times = []
    labels = []
    for index, row in on_events.iterrows():
        ontimes = bs[index].t
        on_event_times.append(ontimes)
        labels.append(row['event'])

    off_event_times = []
    for index, row in off_events.iterrows():
        offtimes = bs[index].t
        off_event_times.append(offtimes)

    #flatten the lists into a single list
    on_event_times = [time for sublist in on_event_times for time in sublist]
    off_event_times = [time for sublist in off_event_times for time in sublist]

    #sort the lists
    on_event_times.sort()
    off_event_times.sort()

    return on_event_times, off_event_times

def get_default_nidaq_map(): #TODO: fold this into parse_nidq
    dinmap = {
        'licks': 0,
        'timing_pulse': 1,
        'solenoid': 2,
        'tone': 3
    }
    ainmap = {
        'photometry': 0,
        'accelerometer_x': 1,
        'accelerometer_y': 2,
        'accelerometer_z': 3,
        'snc': 4,
        'chrimson': 5,
        'chr2': 6
    }

    optoidx = {
        'chrimson': [20, 40, 60, 80, 100, 200],
        'chr2': [20, 40, 60, 80, 100, 200]
    }
    nidaq_map = {
        'dinmap': dinmap,
        'ainmap': ainmap,
        'optoidx': optoidx
    }
    return nidaq_map

