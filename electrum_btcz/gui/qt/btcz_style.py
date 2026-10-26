"""BitcoinZ look and feel: flat, rounded, BTCZ gold accent; light and dark variants."""

from PyQt5.QtGui import QPalette, QColor


LIGHT = {
    'bg': '#f4f5f9',
    'surface': '#ffffff',
    'surface2': '#f8f9fc',
    'border': '#e1e4ec',
    'text': '#1b1e28',
    'muted': '#6b7280',
    'accent': '#e0a516',
    'accent_hover': '#c98f08',
    'accent_text': '#1b1e28',
    'selection': '#fbeecb',
    'scroll': '#c9cdd8',
}

DARK = {
    'bg': '#111319',
    'surface': '#1a1d26',
    'surface2': '#20242f',
    'border': '#2b303d',
    'text': '#e8eaf0',
    'muted': '#9aa1b2',
    'accent': '#f0b429',
    'accent_hover': '#ffc84a',
    'accent_text': '#111319',
    'selection': '#3a3220',
    'scroll': '#3a4050',
}

# the balance header is dark in both themes
HEADER = {
    'bg': '#151821',
    'bg2': '#232838',
    'text': '#ffffff',
    'muted': '#a9b0c2',
    'accent': '#f0b429',
}


def palette(dark: bool) -> QPalette:
    c = DARK if dark else LIGHT
    p = QPalette()
    p.setColor(QPalette.Window, QColor(c['bg']))
    p.setColor(QPalette.WindowText, QColor(c['text']))
    p.setColor(QPalette.Base, QColor(c['surface']))
    p.setColor(QPalette.AlternateBase, QColor(c['surface2']))
    p.setColor(QPalette.Text, QColor(c['text']))
    p.setColor(QPalette.Button, QColor(c['surface']))
    p.setColor(QPalette.ButtonText, QColor(c['text']))
    p.setColor(QPalette.ToolTipBase, QColor(HEADER['bg']))
    p.setColor(QPalette.ToolTipText, QColor(HEADER['text']))
    p.setColor(QPalette.Highlight, QColor(c['accent']))
    p.setColor(QPalette.HighlightedText, QColor(c['accent_text']))
    p.setColor(QPalette.Link, QColor(c['accent_hover'] if not dark else c['accent']))
    p.setColor(QPalette.PlaceholderText, QColor(c['muted']))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor(c['muted']))
    return p


def stylesheet(dark: bool) -> str:
    c = dict(DARK if dark else LIGHT)
    c.update({'h_' + k: v for k, v in HEADER.items()})
    return """
QMainWindow, QDialog, QWizard {{
    background: {bg};
}}

QToolTip {{
    background: {h_bg};
    color: {h_text};
    border: 1px solid {h_bg2};
    border-radius: 6px;
    padding: 6px 8px;
}}

/* balance header */
#btcz_header {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {h_bg}, stop:1 {h_bg2});
}}
#btcz_header QLabel {{
    color: {h_text};
    background: transparent;
}}
#btcz_header_title {{
    font-size: 18px;
    font-weight: 700;
}}
#btcz_header_subtitle, #btcz_header_status {{
    color: {h_muted};
    font-size: 12px;
}}
#btcz_header_balance {{
    color: {h_accent};
    font-size: 26px;
    font-weight: 700;
}}

/* main tabs */
QTabWidget#main_window_nav_bar::pane {{
    border: 0;
    border-top: 1px solid {border};
    background: {surface};
}}
QTabWidget#main_window_nav_bar::tab-bar {{
    left: 12px;
}}
#main_window_nav_bar > QTabBar {{
    background: {bg};
}}
#main_window_nav_bar > QTabBar::tab {{
    background: transparent;
    color: {muted};
    font-size: 14px;
    font-weight: 600;
    padding: 12px 18px 10px 18px;
    margin-right: 4px;
    border: 0;
    border-bottom: 3px solid transparent;
}}
#main_window_nav_bar > QTabBar::tab:hover {{
    color: {text};
}}
#main_window_nav_bar > QTabBar::tab:selected {{
    color: {text};
    border-bottom: 3px solid {accent};
}}

/* other tab widgets (settings, dialogs) */
QTabWidget::pane {{
    border: 1px solid {border};
    border-radius: 8px;
    background: {surface};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {muted};
    padding: 8px 14px;
    border: 0;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{
    color: {text};
    border-bottom: 2px solid {accent};
}}
QTabBar::tab:hover {{
    color: {text};
}}

/* inputs */
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {surface};
    color: {text};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 5px 8px;
    selection-background-color: {accent};
    selection-color: {accent_text};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus {{
    border: 1px solid {accent};
}}
QLineEdit:read-only {{
    background: {surface2};
}}
QComboBox::drop-down {{
    border: 0;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {surface};
    border: 1px solid {border};
    selection-background-color: {selection};
    selection-color: {text};
    outline: 0;
}}
QAbstractItemView QLineEdit {{
    padding: 0px;
    border-radius: 0;
}}

/* buttons */
QPushButton {{
    background: {surface};
    color: {text};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 7px 16px;
    min-height: 18px;
}}
QPushButton:hover {{
    border: 1px solid {accent};
}}
QPushButton:pressed {{
    background: {selection};
}}
QPushButton[primary="true"], QPushButton:default {{
    background: {accent};
    color: {accent_text};
    border: 1px solid {accent};
    font-weight: 600;
}}
QPushButton[primary="true"]:hover, QPushButton:default:hover {{
    background: {accent_hover};
    border: 1px solid {accent_hover};
}}
QPushButton:disabled {{
    color: {muted};
    background: {surface2};
    border: 1px solid {border};
}}
QPushButton:flat {{
    border: 0;
    background: transparent;
}}
QToolButton {{
    background: transparent;
    border: 0;
    border-radius: 6px;
    padding: 3px;
}}
QToolButton:hover {{
    background: {selection};
}}

/* lists and tables */
QTreeView, QTableView, QListView, QTreeWidget, QListWidget {{
    background: {surface};
    alternate-background-color: {surface2};
    border: 0;
    outline: 0;
    selection-background-color: {selection};
    selection-color: {text};
}}
QTreeView::item, QTableView::item, QListView::item {{
    padding: 5px 4px;
    border: 0;
}}
QTreeView::item:selected, QTableView::item:selected, QListView::item:selected {{
    background: {selection};
    color: {text};
}}
QTreeView::item:hover {{
    background: {surface2};
}}
QHeaderView {{
    background: {surface};
    border: 0;
}}
QHeaderView::section {{
    background: {surface};
    color: {muted};
    font-weight: 600;
    border: 0;
    border-bottom: 1px solid {border};
    padding: 8px 6px;
}}

/* scroll bars */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {scroll};
    border-radius: 3px;
    min-height: 30px;
    min-width: 30px;
}}
QScrollBar::handle:hover {{
    background: {muted};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0;
    height: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

/* menus */
QMenuBar {{
    background: {bg};
    color: {text};
}}
QMenuBar::item {{
    background: transparent;
    padding: 4px 10px;
}}
QMenuBar::item:selected {{
    background: {selection};
    border-radius: 6px;
}}
QMenu {{
    background: {surface};
    color: {text};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 5px;
}}
QMenu::item {{
    padding: 6px 22px 6px 14px;
    border-radius: 6px;
}}
QMenu::item:selected {{
    background: {selection};
    color: {text};
}}
QMenu::separator {{
    height: 1px;
    background: {border};
    margin: 4px 8px;
}}

/* status bar */
QStatusBar {{
    background: {surface};
    border-top: 1px solid {border};
    color: {muted};
}}
QStatusBar::item {{
    border: 0;
}}
QStatusBar QLabel {{
    color: {muted};
    padding: 0 6px;
}}

/* misc */
QGroupBox {{
    border: 1px solid {border};
    border-radius: 10px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {muted};
}}
QProgressBar {{
    background: {surface2};
    border: 1px solid {border};
    border-radius: 6px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {accent};
    border-radius: 5px;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {border};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {accent};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {accent};
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QScrollArea {{
    border: 0;
    background: transparent;
}}
QSplitter::handle {{
    background: {border};
}}
""".format(**c)
