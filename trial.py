s = """settings.maxIterations = 100;
        settings.sizeX = 500;
        settings.sizeY = 500;
        settings.transformOperators.add(-5);
        settings.transformOperators.add(-6);
        settings.transformOperators.add(-1);
        settings.leftest = -16;
        settings.rightest = 16;
        settings.highest = 16;
        settings.lowest = -16;"""

s = s.replace("settings.", "s.").replace("maxIterations", "max_iter").replace("sizeX",
                                                                              "width").replace(
    "sizeY", "height").replace("highest", "top").replace("lowest", "bottom").replace("est", "")
s = s.replace(";", "")
print(s)