from f1sim.viewer.console.app import create_app
from f1sim.viewer.console.training_setup import TrainingSetupForm
app=create_app(['solo-panel'])
f=TrainingSetupForm();f.resize(1050,760);f.set_mode('dagger');f.set_values({'solo_fraction':.3,'race_size':2,'envs':4,'teacher_kind':'interactive','action_mode':'plan'});f.search.setText('solo');f.show();app.processEvents();f.grab().save('work/dagger-solo-panel/solo-fields.png');f.close()
