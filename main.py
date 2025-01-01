import openai
from pydantic import BaseModel
import random
import re
import time
from copy import copy as duplicate
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
import os
from dotenv import load_dotenv
import sys
import argparse

# Add after imports but before other code
def parse_args():
    parser = argparse.ArgumentParser(description='Generate a crossword puzzle from a given subject')
    parser.add_argument('subject', type=str, help='The subject for the crossword puzzle')
    return parser.parse_args()

# Load environment variables from .env file
load_dotenv()

# Create PDFs directory if it doesn't exist
pdf_dir = "PDFs"
os.makedirs(pdf_dir, exist_ok=True)

# ========================================================================
# 0) OpenAI PROMPT & CROSSWORD FETCH
# ========================================================================

def main():
    args = parse_args()
    PROMPT_SUBJECT = args.subject  # Get subject from command line argument
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    openai.api_key = OPENAI_API_KEY

    class CrosswordData(BaseModel):
        """Structure for crossword puzzle data from OpenAI."""
        title: str
        words: list[str] 
        clues: list[str]

    # Make a call to OpenAI
    completion = openai.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Produce a clever crossword title and 30 high quality, professional, sometimes humourous, crossword words and clues based on the subject.Do not number the clues."
                ),
            },
            {
                "role": "user",
                "content": f"The subject is {PROMPT_SUBJECT}",
            },
        ],
        response_format=CrosswordData,
    )

    event = completion.choices[0].message.parsed

    # Extract the title, words, and clues from the OpenAI response
    puzzle_title = event.title.strip()
    word_data = list(zip(event.words, event.clues))

    # ========================================================================
    # 1) CROSSWORD CLASSES
    # ========================================================================

    class Word(object):
        def __init__(self, word=None, clue=None):
            # Remove spaces, lower-case
            self.word = re.sub(r'\s+', '', word.lower())
            self.clue = clue
            self.length = len(self.word)
            self.row = None
            self.col = None
            self.vertical = None
            self.number = None

        def down_across(self):
            return 'down' if self.vertical else 'across'

        def __repr__(self):
            return self.word

    class Crossword(object):
        def __init__(self, cols, rows, empty='-', maxloops=2000, available_words=[]):
            self.cols = cols
            self.rows = rows
            self.empty = empty
            self.maxloops = maxloops
            self.available_words = available_words
            self.randomize_word_list()
            self.current_word_list = []
            self.clear_grid()
            self.debug = 0

        def clear_grid(self):
            self.grid = []
            for _ in range(self.rows):
                row = [self.empty]*self.cols
                self.grid.append(row)

        def randomize_word_list(self):
            temp_list = []
            for w in self.available_words:
                if isinstance(w, Word):
                    temp_list.append(Word(w.word, w.clue))
                else:
                    temp_list.append(Word(w[0], w[1]))
            random.shuffle(temp_list)
            # Sort by length descending
            temp_list.sort(key=lambda i: len(i.word), reverse=True)
            self.available_words = temp_list

        def compute_crossword(self, time_permitted=1.00, spins=2):
            copy_x = Crossword(self.cols, self.rows, self.empty, self.maxloops, self.available_words)

            count = 0
            start_time = time.time()
            while (time.time() - start_time) < time_permitted or count == 0:
                self.debug += 1
                copy_x.randomize_word_list()
                copy_x.current_word_list = []
                copy_x.clear_grid()

                x = 0
                while x < spins:
                    for w in copy_x.available_words:
                        if w not in copy_x.current_word_list:
                            copy_x.fit_and_add(w)
                    x += 1

                if len(copy_x.current_word_list) > len(self.current_word_list):
                    self.current_word_list = copy_x.current_word_list
                    self.grid = copy_x.grid
                count += 1

        def suggest_coord(self, word):
            coordlist = []
            glc = -1
            for given_letter in word.word:
                glc += 1
                rowc = 0
                for row in self.grid:
                    rowc += 1
                    colc = 0
                    for cell in row:
                        colc += 1
                        if given_letter == cell:
                            # vertical
                            try:
                                if rowc - glc > 0:
                                    if ((rowc - glc) + word.length) <= self.rows:
                                        coordlist.append([colc, rowc-glc, 1, 0, 0])
                            except:
                                pass
                            # horizontal
                            try:
                                if colc - glc > 0:
                                    if ((colc - glc) + word.length) <= self.cols:
                                        coordlist.append([colc-glc, rowc, 0, 0, 0])
                            except:
                                pass
            return self.sort_coordlist(coordlist, word)

        def sort_coordlist(self, coordlist, word):
            new_coordlist = []
            for coord in coordlist:
                col, row, vertical = coord[0], coord[1], coord[2]
                score = self.check_fit_score(col, row, vertical, word)
                coord[4] = score
                if score > 0:
                    new_coordlist.append(coord)
            random.shuffle(new_coordlist)
            new_coordlist.sort(key=lambda i: i[4], reverse=True)
            return new_coordlist

        def fit_and_add(self, word):
            fit = False
            count = 0
            coordlist = self.suggest_coord(word)

            while not fit and count < self.maxloops:
                if len(self.current_word_list) == 0:
                    # seed
                    vertical, col, row = random.randrange(0,2), 1, 1
                    if self.check_fit_score(col, row, vertical, word):
                        fit = True
                        self.set_word(col, row, vertical, word, force=True)
                else:
                    try:
                        col, row, vertical = coordlist[count][0], coordlist[count][1], coordlist[count][2]
                    except IndexError:
                        return
                    if coordlist[count][4]:
                        fit = True
                        self.set_word(col, row, vertical, word, force=True)
                count += 1

        def check_fit_score(self, col, row, vertical, word):
            if col < 1 or row < 1:
                return 0
            count, score = 1, 1
            for letter in word.word:
                try:
                    active_cell = self.get_cell(col, row)
                except IndexError:
                    return 0
                if active_cell not in (self.empty, letter):
                    return 0
                if active_cell == letter:
                    score += 1

                # surroundings
                if vertical:
                    if active_cell != letter:
                        if not self.check_if_cell_clear(col+1, row): return 0
                        if not self.check_if_cell_clear(col-1, row): return 0
                    if count == 1:
                        if not self.check_if_cell_clear(col, row-1): return 0
                    if count == len(word.word):
                        if not self.check_if_cell_clear(col, row+1): return 0
                else:
                    if active_cell != letter:
                        if not self.check_if_cell_clear(col, row-1): return 0
                        if not self.check_if_cell_clear(col, row+1): return 0
                    if count == 1:
                        if not self.check_if_cell_clear(col-1, row): return 0
                    if count == len(word.word):
                        if not self.check_if_cell_clear(col+1, row): return 0

                if vertical:
                    row += 1
                else:
                    col += 1
                count += 1
            return score

        def set_word(self, col, row, vertical, word, force=False):
            if force:
                word.col = col
                word.row = row
                word.vertical = vertical
                self.current_word_list.append(word)
                for letter in word.word:
                    self.set_cell(col, row, letter)
                    if vertical:
                        row += 1
                    else:
                        col += 1

        def set_cell(self, col, row, value):
            self.grid[row-1][col-1] = value

        def get_cell(self, col, row):
            return self.grid[row-1][col-1]

        def check_if_cell_clear(self, col, row):
            try:
                if self.get_cell(col, row) == self.empty:
                    return True
            except IndexError:
                pass
            return False

        def solution(self):
            outStr = ""
            for r in range(self.rows):
                for c in self.grid[r]:
                    outStr += c + " "
                outStr += "\n"
            return outStr

    # ========================================================================
    # 2) BUILD THE PUZZLE
    # ========================================================================

    puzzle = Crossword(
        cols=20,
        rows=20,
        empty='-',
        maxloops=5000,
        available_words=word_data
    )
    puzzle.compute_crossword(time_permitted=2)

    # ========================================================================
    # 3) NUMBER THE WORD STARTS
    # ========================================================================

    def number_crossword_squares(puzzle):
        """
        Numbers squares according to standard crossword convention:
        - Numbers increment from left to right, top to bottom
        - Only number squares that start at least one word
        - Each square gets only one number, even if it starts both across and down words
        """
        numbering = {}
        next_num = 1
        
        # Scan grid from top to bottom, left to right
        for r in range(puzzle.rows):
            for c in range(puzzle.cols):
                # Skip empty squares
                if puzzle.grid[r][c] == puzzle.empty:
                    continue
                    
                # Check if this square starts any word
                starts_word = False
                
                # Check if it starts an across word
                is_across_start = (
                    # Must have empty square or grid edge to left
                    (c == 0 or puzzle.grid[r][c-1] == puzzle.empty) and
                    # Must have at least one letter to right
                    (c < puzzle.cols-1 and puzzle.grid[r][c+1] != puzzle.empty)
                )
                
                # Check if it starts a down word
                is_down_start = (
                    # Must have empty square or grid edge above
                    (r == 0 or puzzle.grid[r-1][c] == puzzle.empty) and
                    # Must have at least one letter below
                    (r < puzzle.rows-1 and puzzle.grid[r+1][c] != puzzle.empty)
                )
                
                # If this square starts either type of word, number it
                if is_across_start or is_down_start:
                    numbering[(r, c)] = next_num
                    next_num += 1
        
        return numbering

    numbering_dict = number_crossword_squares(puzzle)
    for w in puzzle.current_word_list:
        r0 = w.row - 1
        c0 = w.col - 1
        w.number = numbering_dict[(r0, c0)]

    # Separate across vs down
    across_clues = []
    down_clues = []
    for w in puzzle.current_word_list:
        if w.down_across() == 'across':
            across_clues.append((w.number, w.clue))
        else:
            down_clues.append((w.number, w.clue))

    across_clues.sort(key=lambda x: x[0])
    down_clues.sort(key=lambda x: x[0])

    # ========================================================================
    # 4) CREATE THE TWO-PAGE PDF
    # ========================================================================

    def wrap_text(text, max_chars=52):
        """Simple word-wrap."""
        words = text.split()
        lines = []
        current_line = []
        length_so_far = 0
        for w in words:
            if length_so_far + len(w) + len(current_line) > max_chars:
                lines.append(" ".join(current_line))
                current_line = [w]
                length_so_far = len(w)
            else:
                current_line.append(w)
                length_so_far += len(w)
        if current_line:
            lines.append(" ".join(current_line))
        return lines

    def create_crossword_pdfs(
        puzzle,
        numbering,
        across_clues,
        down_clues,
        title="My Crossword Puzzle"
    ):
        # Replace spaces with underscores for filename
        safe_title = re.sub(r"\s+", "_", title.strip())
        # Create separate filenames for puzzle and solution
        puzzle_filename = os.path.join(pdf_dir, f"{safe_title}_puzzle.pdf")
        solution_filename = os.path.join(pdf_dir, f"{safe_title}_solution.pdf")

        def draw_page(canvas_obj, include_answers=False):
            page_width, page_height = landscape(A4)
            left_margin = 50
            top_margin = page_height - 50
            text_width = 360

            # Puzzle title
            canvas_obj.setFont("Helvetica-Bold", 16)
            canvas_obj.drawString(left_margin, top_margin, title)
            if include_answers:
                canvas_obj.drawString(left_margin + 450, top_margin, "SOLUTION")

            y_text = top_margin - 30

            # ACROSS
            canvas_obj.setFont("Helvetica-Bold", 12)
            canvas_obj.drawString(left_margin, y_text, "Across")
            y_text -= 15
            canvas_obj.setFont("Helvetica", 8)
            for num, clue in across_clues:
                lines = wrap_text(f"{num}. {clue}", max_chars=60)
                for ln in lines:
                    canvas_obj.drawString(left_margin, y_text, ln)
                    y_text -= 10
                y_text -= 2

            y_text -= 8

            # DOWN
            canvas_obj.setFont("Helvetica-Bold", 12)
            canvas_obj.drawString(left_margin, y_text, "Down")
            y_text -= 15
            canvas_obj.setFont("Helvetica", 8)
            for num, clue in down_clues:
                lines = wrap_text(f"{num}. {clue}", max_chars=60)
                for ln in lines:
                    canvas_obj.drawString(left_margin, y_text, ln)
                    y_text -= 10
                y_text -= 2

            # Draw the puzzle grid
            cell_size = 20
            puzzle_left = left_margin + text_width - 30  # Reduced by 60 pixels
            puzzle_top = top_margin - 30  # Moved down

            canvas_obj.setLineWidth(1)
            for r in range(puzzle.rows):
                for cc in range(puzzle.cols):
                    x = puzzle_left + cc * cell_size
                    y = puzzle_top - r * cell_size
                    
                    # Draw black squares for empty cells
                    if puzzle.grid[r][cc] == puzzle.empty:
                        canvas_obj.setFillColor('black')
                        canvas_obj.rect(x, y - cell_size, cell_size, cell_size, fill=1)
                    else:
                        canvas_obj.setFillColor('white')
                        canvas_obj.rect(x, y - cell_size, cell_size, cell_size, fill=1)
                        
                        # Add numbers if present
                        if (r, cc) in numbering:
                            canvas_obj.setFillColor('black')
                            canvas_obj.setFont("Helvetica", 6)
                            canvas_obj.drawString(x + 2, (y - cell_size) + 10, str(numbering[(r, cc)]))
                        
                        # Add letters if this is the solution page
                        if include_answers:
                            canvas_obj.setFillColor('black')
                            canvas_obj.setFont("Helvetica-Bold", 10)
                            letter = puzzle.grid[r][cc].upper()
                            canvas_obj.drawCentredString(x + cell_size/2, (y - cell_size) + cell_size/2 - 2, letter)

        # Create puzzle PDF
        c_puzzle = canvas.Canvas(puzzle_filename, pagesize=landscape(A4))
        draw_page(c_puzzle, include_answers=False)
        c_puzzle.save()

        # Create solution PDF
        c_solution = canvas.Canvas(solution_filename, pagesize=landscape(A4))
        draw_page(c_solution, include_answers=True)
        c_solution.save()

        print(f"Crossword puzzle created: {puzzle_filename}")
        print(f"Solution created: {solution_filename}")

    # Update the call to the PDF creation function
    create_crossword_pdfs(
        puzzle=puzzle,
        numbering=numbering_dict,
        across_clues=across_clues,
        down_clues=down_clues,
        title=puzzle_title
    )

    # Print solution to console (optional)
    print("Crossword solution (letters placed):")
    print(puzzle.solution())

if __name__ == "__main__":
    main()