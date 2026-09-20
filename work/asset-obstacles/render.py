from PyQt5 import QtWidgets
from f1sim.viewer.console.app import create_app
from f1sim.viewer.console.training import TrackPicker
from f1sim.viewer.console.window import ConsoleWindow
app=create_app(['assets'])
p=TrackPicker();p.resize(900,690);p.show();app.processEvents();p.grab().save('work/asset-obstacles/training-picker.png')
w=ConsoleWindow();w.resize(1300,900);w._selected_map='real/icra22';w._sync_obstacle_options();w._on_scenario_changed();w.show();app.processEvents();w.combo_obstacle.showPopup();app.processEvents();w.combo_obstacle.view().window().grab().save('work/asset-obstacles/driving-choices.png');w.combo_obstacle.hidePopup();w.allow_close();p.close()
