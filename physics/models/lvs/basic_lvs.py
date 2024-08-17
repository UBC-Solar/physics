from physics.models.lvs.base_lvs import BaseLVS


class BasicLVS(BaseLVS):

    def __init__(self, consumed_energy, lvs_current, lvs_voltage):
        super().__init__(consumed_energy)
        self.lvs_current = lvs_current
        self.lvs_voltage = lvs_voltage

    def get_consumed_energy(self, tick):
        """
            Get the energy consumption of the Low Voltage System (current * voltage * time)

            :param tick - (int) tick time passed
            :returns: consumed_energy - (number) value of energy consumed
        """
        return self.lvs_current * self.lvs_voltage * tick
