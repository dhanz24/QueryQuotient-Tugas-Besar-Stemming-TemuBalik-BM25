import sys
import os
import PyPDF2
from docx import Document
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog,QMessageBox
from PyQt5.QtCore import QFile, QTextStream
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon
import math


# Download sumber daya NLTK yang diperlukan

class ProjectBesar(QMainWindow):
    def __init__(self):
        super(ProjectBesar, self).__init__()
        loadUi('gui.ui', self)
        style_file_path = 'Perstfic.qss'  # Make sure the file name is correct
        style_file = QFile(style_file_path)

        if style_file.open(QFile.ReadOnly | QFile.Text):
            stream = QTextStream(style_file)
            stylesheet = stream.readAll()
            self.setStyleSheet(stylesheet)
            style_file.close()
        else:
            print(f"Failed to open stylesheet file: {style_file_path}")
        self.setWindowTitle("QueryQuotient")
        self.setWindowIcon(QIcon("logo.png"))
        self.pushButton.clicked.connect(self.search)
        self.file.clicked.connect(self.display_file_list)
        self.comboBox.currentIndexChanged.connect(self.load_selected_file)
        self.query_text = ""
        self.documents_directory = 'D:\\Project Datmin\\Files'
        self.files = []
        self.documents = []
        self.tokenized_documents = []
        self.selected_file_index = -1
        self.pushButton_2.clicked.connect(self.delete_document)
        self.doc_length = 0
        self.avg_doc_length = 0.0
        self.idf_values = {}
        self.tf_values = {}
        self.load_files()
    
    def display_file_list(self):
        self.directory_path = self.textEdit.toPlainText()
        files_list = list_files_in_directory(self.directory_path)
        self.list_file.clear()
        
        self.list_file.append("List File dalam Direktori:")
        for index, file_name in enumerate(files_list, start=1):
            self.list_file.append(f"{index}. {file_name}")

    def load_files(self):
        self.files = [f for f in os.listdir(self.documents_directory) if os.path.isfile(os.path.join(self.documents_directory, f))]
        self.comboBox.addItems(self.files)

        # Memuat dan memproses semua dokumen
        self.documents = []
        self.tokenized_documents = []
        for file_name in self.files:
            file_path = os.path.join(self.documents_directory, file_name)
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            if ext == '.pdf':
                text = self.read_pdf(file_path)
            elif ext == '.docx':
                text = self.read_docx(file_path)
            else:
                text = self.read_txt(file_path)

            self.documents.append(text)
            self.tokenized_documents.append(self.preprocess_text(text))

    def read_pdf(self, file_path):
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            return text
    
    def read_docx(self, file_path):
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + " "
        return text
    
    def read_txt(self, file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()

    def load_selected_file(self):
        self.selected_file_index = self.comboBox.currentIndex()
        
        if self.selected_file_index != -1:
            file_path = os.path.join(self.documents_directory, self.files[self.selected_file_index])
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            if ext == '.pdf':
                text = self.read_pdf(file_path)
            elif ext == '.docx':
                text = self.read_docx(file_path)
            else:
                text = self.read_txt(file_path)

            self.textBrowser.setText(text)

    def preprocess_text(self, text):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        text = stopword_remover.remove(text)
        tokens = nltk.word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalnum()]
        tokens = [stemmer.stem(word) for word in tokens]

        return tokens

    def calculate_idf(self, term):
        doc_freq = sum(1 for doc_tokens in self.tokenized_documents if term in doc_tokens)
        return math.log((len(self.tokenized_documents) - doc_freq + 0.5) / (doc_freq + 0.5) )

    def calculate_bm25_score(self, query_tokens, doc_tokens):
        score = 0
        k1 = 1.5
        b = 0.75
        self.avg_doc_length = sum(len(doc) for doc in self.tokenized_documents) / len(self.tokenized_documents)
        self.doc_length = len(doc_tokens)

        for term in query_tokens:
            if term in doc_tokens:
                df = sum(1 for doc_tokens in self.tokenized_documents if term in doc_tokens)
                idf = self.calculate_idf(term)
                self.idf_values[term] = idf  # Menyimpan nilai idf
                tf = doc_tokens.count(term)
                self.tf_values[term] = tf  # Menyimpan nilai tf
                score += idf * (tf * (k1 + 1)) / (k1 * ((1 - b) + b * (self.doc_length / self.avg_doc_length))+tf)

        return score

    def rank_documents_bm25(self, query_tokens):
        scores = [(i, self.calculate_bm25_score(query_tokens, doc_tokens)) for i, doc_tokens in enumerate(self.tokenized_documents)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def perform_search(self):
        if self.selected_file_index != -1:
            query = self.query_text
            tokenized_query = self.preprocess_text(query)

            # Rank documents based on BM25 scores
            document_scores = self.rank_documents_bm25(tokenized_query)

            # Display search results for all documents
            search_results = "\nBM25 - Hasil Pencarian untuk Semua Dokumen:\n"

            for i, (doc_index, score) in enumerate(document_scores, 1):
                document_name = self.files[doc_index]
                search_results += f"\nRank {i}: Dokumen '{document_name}':\n"
                search_results += f"Similarity Score: {score:.4f}\n"

                # Display additional information for each ranked document
                similarity_text = f"\nSimilarity Score untuk Dokumen '{document_name}':\n"
                similarity_text += f"Similarity Score: {score:.4f}\n"

                similarity_text += f"Doc Length: {len(self.tokenized_documents[doc_index])}\n"
                similarity_text += f"Avg Doc Length: {self.avg_doc_length:.2f}\n"
                
                similarity_text += "\nIDF Values:\n"
                for term, idf_value in self.idf_values.items():
                    similarity_text += f"{term}: {idf_value:.4f}\n"
                
                similarity_text += "\nTF Values:\n"
                for term, tf_value in self.tf_values.items():
                    similarity_text += f"{term}: {tf_value}\n"

                # Display stemmed words and counts for each document
                if doc_index == self.selected_file_index:
                    similarity_text += "\n\nDetails for Selected Document:\n"

                    # Display case folding
                    case_folded_text = [word.lower() for word in nltk.word_tokenize(self.documents[doc_index]) if word.isalnum()]
                    similarity_text += f"\nCase Folding:\n{' '.join(case_folded_text)}\n"
                    self.textBrowser_3.setText(" ".join(case_folded_text))

                    # Display tokenization
                    tokenized_text = [word for word in nltk.word_tokenize(self.documents[doc_index]) if word.isalnum()]
                    similarity_text += f"\nTokenization:\n{' '.join(tokenized_text)}\n"
                    self.textBrowser_4.setText(" ".join(tokenized_text))

                    # Display stemming
                    stemmed_words_count = self.display_stemmed_words_and_count(self.documents[doc_index])
                    similarity_text += f"\nStemming:\n{stemmed_words_count}"
                    self.textBrowser_5.setText(stemmed_words_count)

                    # Display preprocessed text
                    preprocessed_text = self.preprocess_text(self.documents[doc_index])
                    similarity_text += f"\nPreprocessed Text:\n{' '.join(preprocessed_text)}"
                    self.textBrowser_6.setText(" ".join(preprocessed_text))

                current_text = self.textBrowser_2.toPlainText()
                self.textBrowser_2.setText(current_text + similarity_text)

            # Display search results in the main textBrowser
            self.textBrowser_1.setText(search_results)                  

            # Continue with displaying details for the selected document as before...

            selected_document = self.files[self.selected_file_index]
            similarity_score = self.calculate_bm25_score(tokenized_query, self.tokenized_documents[self.selected_file_index])

            # Tampilkan hasil pencarian
            search_results_selected = f"\nBM25 - Hasil Pencarian untuk Dokumen '{selected_document}':\n"

            if similarity_score is not None:
                search_results_selected += f"Similarity Score: {similarity_score:.4f}\n"
            else:
                search_results_selected += "Dokumen tidak ditemukan dalam hasil peringkat.\n"

            # ... (rest of the code for displaying details for the selected document)   


    def closeEvent(self, event):
        local_msg_box = QMessageBox()
        local_msg_box.setWindowIcon(QIcon("logo.png"))
        local_msg_box.setIcon(QMessageBox.Question)
        local_msg_box.setWindowTitle("Konfirmasi")
        local_msg_box.setText("Apakah Anda yakin ingin keluar dari aplikasi?")
        local_msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        
        # Set the stylesheet for the QMessageBox
        local_msg_box.setStyleSheet("""
            QLabel {
                color: #2c2c2c;
            }
            QPushButton {
                background-color: qlineargradient(spread:pad, x1:1, y1:0, x2:1, y2:1, stop:0 rgba(42, 95, 113, 255),stop:1 rgba(12, 53, 97, 255));
                color: #fff;
            }
            QPushButton:hover {
                background-color: qlineargradient(spread:pad, x1:1, y1:0, x2:1, y2:1, stop:0 rgba(167, 214, 50, 255),stop:1 rgba(98, 169, 67, 255));
                color: #fff;
            }
            QPushButton::pressed{
                background-color: qlineargradient(spread:pad, x1:1, y1:0, x2:1, y2:1, stop:0 rgba(147, 194, 30, 255),stop:1 rgba(78, 149, 47, 255));
                color: #fff;
            }
        """)


        # Display the QMessageBox and get the user's response
        reply = local_msg_box.exec_()

        if reply == QMessageBox.Yes:
            # Continue with the application exit
            event.accept()
        else:
            # Ignore the close event
            event.ignore()
            pass



    def display_stemmed_words_and_count(self, document):
        words = self.preprocess_text(document)
        word_count = {word: words.count(word) for word in set(words)}

        stemmed_results = "Jumlah Kata yang di Stemming : \n"
        for word, count in word_count.items():
            stemmed_results += f"{word}: {count}\n"

        return stemmed_results if stemmed_results else "Tidak ada kata yang diproses\n"

    def search(self):
        self.query_text = self.textEdit.toPlainText()
        self.perform_search()


    def delete_document(self):
        if self.selected_file_index != -1:
            # Membuat QMessageBox lokal
            local_msg_box = QMessageBox()
            local_msg_box.setWindowIcon(QIcon("logo.png"))
            local_msg_box.setIcon(QMessageBox.Question)
            local_msg_box.setWindowTitle("Konfirmasi")
            local_msg_box.setText("Apakah Anda yakin ingin menghapus hasil proses ?")
            local_msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            # Mengatur gaya khusus untuk QMessageBox ini
            local_msg_box.setStyleSheet("""
               
                QLabel {
                    color: #2c2c2c;
                }
                QPushButton {
                    background-color: qlineargradient(spread:pad, x1:1, y1:0, x2:1, y2:1, stop:0 rgba(42, 95, 113, 255),stop:1 rgba(12, 53, 97, 255));
                    color: #fff;
                    
            
                }
                QPushButton:hover {
                   background-color: qlineargradient(spread:pad, x1:1, y1:0, x2:1, y2:1, stop:0 rgba(167, 214, 50, 255),stop:1 rgba(98, 169, 67, 255));
	               color: #fff;
                   
                }
                QPushButton::pressed{
                background-color: qlineargradient(spread:pad, x1:1, y1:0, x2:1, y2:1, stop:0 rgba(147, 194, 30, 255),stop:1 rgba(78, 149, 47, 255));
	            color: #fff; 
             
                }   
            """)

            # Menampilkan QMessageBox dan mendapatkan jawaban
            reply = local_msg_box.exec_()

            if reply == QMessageBox.Yes:
                selected_document = self.files[self.selected_file_index]

                # Bersihkan tampilan
                self.textBrowser_1.clear()
                self.textBrowser_2.clear()
                self.textBrowser_3.clear()
                self.textBrowser_4.clear()
                self.textBrowser_5.clear()
                self.textBrowser_6.clear()

                # Reset indeks dokumen terpilih
                self.selected_file_index = -1
            else:
                # Pengguna membatalkan penghapusan
                pass
def list_files_in_directory(directory):
    files = os.listdir(directory)
    return files  

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProjectBesar()
    window.show()
    sys.exit(app.exec_())
