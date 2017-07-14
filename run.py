# -*- coding: UTF-8 -*-

"""
Entry point of the app
"""

from app import app

try:
    app.run(port='5005', use_reloader=False, debug=True)
except:
    print("Some thing wrong!")