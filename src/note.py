from typing import Self, overload

from pydantic import BaseModel, Field

REF_NOTE = 9,4 # A4
REF_FREQ = 440 #Hz


class Note(BaseModel):
    NoteNumber: int = Field(0, ge=0, lt=12)
    Octave: int = Field(5)
    sharp: bool = Field(True)

    # Arithmetic
    def __add__(self, other: int) -> Self:
        final = self.NoteNumber + 12 * self.Octave + other
        return type(self)(
            NoteNumber=final % 12,
            Octave=final // 12,
            sharp=self.sharp,
        )

    @overload
    def __sub__(self, other: int) -> Self:...
    
    @overload
    def __sub__(self,other:Self) ->int:...

    def __sub__(self, other: int | Self) -> Self | int:
        self_position = self.NoteNumber + self.Octave * 12

        if isinstance(other, Note):
            other_position = other.NoteNumber + other.Octave * 12
            return self_position - other_position

        if isinstance(other, int):
            final = self_position - other
            return type(self)(
                NoteNumber=final % 12,
                Octave=final // 12,
                sharp=self.sharp,
            )

        return NotImplemented

    # Name as a string
    @property
    def name(self) -> str:
        return f"{self.__number_to_letter(self.NoteNumber, self.sharp)}{self.Octave}"
    
    @staticmethod
    def __number_to_letter(n: int, sharp: bool = True) -> str:
        note_names = (
            ("C",),
            ("C#", "Db"),
            ("D",),
            ("D#", "Eb"),
            ("E",),
            ("F",),
            ("F#", "Gb"),
            ("G",),
            ("G#", "Ab"),
            ("A",),
            ("A#", "Bb"),
            ("B",),
        )

        if not 0 <= n < len(note_names):
            raise ValueError("note number must be between 0 and 11")

        names = note_names[n]
        return names[0] if sharp or len(names) == 1 else names[1]

    # Frequency
    @property
    def frequency(self)->float:
        # Distance from A4
        d = self.NoteNumber + 12*self.Octave - REF_NOTE[0] - 12* REF_NOTE[1]
        return REF_FREQ*2**(d/12)