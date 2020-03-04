from takahe import BinaryStarSystem

class BinaryStarSystemEnsemble:
    """Represents a collection of binary star systems.

    Represents a group of binary star system objects. Will be
    generated if the loader encounters a group of BSS objects (e.g.,
    from BPASS).
    """
    def __init__(self):
        self.__ensemble = []
        self.__count = 0
        self.__pointer = 0

    def add(self, binary_star):
        """Add a BSS to the current ensemble.

        Arguments:
            binary_star {BinaryStarSystem} -- The BSS to add.

        Raises:
            TypeError -- If the Binary Star System is not an instance of
                         BinaryStarSystem.BinaryStarSystem.
        """
        if type(binary_star) != BinaryStarSystem.BinaryStarSystem:
            raise TypeError("binary_star must be an instance of BinaryStarSystem!")

        self.__ensemble.append(binary_star)
        self.__count += 1

    def merge_rate(self, t_merge, return_as="rel"):
        count = 0

        if return_as.lower() not in ['abs', 'rel']:
            raise ValueError("return_as must be either abs or rel")

        for i in range(self.__count):
            binary_star = self.__ensemble[i]
            valid = (binary_star.coalescence_time() <= t_merge)
            if valid:
                count += 1

        if return_as == 'abs':
            return count
        elif return_as == 'rel':
            return count / self.__count

    def __len__(self):
        return self.__count

    def __iter__(self):
        return self

    def __next__(self):
        if self.__pointer >= self.__count:
            raise StopIteration
        else:
            self.__pointer += 1
            return self.__ensemble[self.__pointer - 1]
