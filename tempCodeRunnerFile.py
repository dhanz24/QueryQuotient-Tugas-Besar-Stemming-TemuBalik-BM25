def closeEvent(self, event):
            reply = QMessageBox.question(self, 'Konfirmasi', 'Apakah Anda yakin ingin keluar dari aplikasi?',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()