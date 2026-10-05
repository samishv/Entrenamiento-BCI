from psychopy import prefs
prefs.hardware['audioLib'] = ['ptb']
from psychopy import sound
from psychtoolbox import audio   # import explícito del submódulo

for d in audio.get_devices():
    if d['NrOutputChannels'] > 0:
        print(d['DeviceIndex'], '|', d['DeviceName'], '|', d['HostAudioAPIName'])