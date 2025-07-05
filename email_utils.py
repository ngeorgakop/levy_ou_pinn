import smtplib
import ssl
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Optional, List
import logging

from config import (
    EMAIL_HOST, EMAIL_PORT, EMAIL_HOST_USER, 
    EMAIL_HOST_PASSWORD, EMAIL_USE_TLS
)

logger = logging.getLogger(__name__)

def send_email(
    to_emails: List[str],
    subject: str,
    message: str,
    html_message: Optional[str] = None
) -> bool:
    """
    Send an email using configured SMTP settings.
    
    Args:
        to_emails: List of recipient email addresses
        subject: Email subject
        message: Plain text message
        html_message: Optional HTML message
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    if not EMAIL_HOST_USER or not EMAIL_HOST_PASSWORD:
        logger.warning("Email credentials not configured. Cannot send email.")
        return False
    
    try:
        # Create message
        msg = MimeMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_HOST_USER
        msg["To"] = ", ".join(to_emails)
        
        # Add text part
        text_part = MimeText(message, "plain")
        msg.attach(text_part)
        
        # Add HTML part if provided
        if html_message:
            html_part = MimeText(html_message, "html")
            msg.attach(html_part)
        
        # Create SMTP session
        if EMAIL_USE_TLS:
            server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
            server.starttls()  # Enable TLS
        else:
            server = smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT)
        
        # Login and send email
        server.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
        server.sendmail(EMAIL_HOST_USER, to_emails, msg.as_string())
        server.quit()
        
        logger.info(f"Email sent successfully to {to_emails}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        return False

def notify_training_complete(loss_history: List[float], admin_email: str) -> bool:
    """
    Send notification when training is complete.
    
    Args:
        loss_history: List of loss values during training
        admin_email: Email address to send notification to
        
    Returns:
        bool: True if notification sent successfully
    """
    final_loss = loss_history[-1] if loss_history else "N/A"
    
    subject = "PINN Training Complete"
    message = f"""
    Your Lévy-driven OU PINN training has completed successfully!
    
    Final Loss: {final_loss}
    Total Epochs: {len(loss_history)}
    
    Check your workspace for the trained model.
    """
    
    html_message = f"""
    <html>
    <body>
        <h2>🎉 PINN Training Complete</h2>
        <p>Your Lévy-driven OU PINN training has completed successfully!</p>
        <ul>
            <li><strong>Final Loss:</strong> {final_loss}</li>
            <li><strong>Total Epochs:</strong> {len(loss_history)}</li>
        </ul>
        <p>Check your workspace for the trained model.</p>
    </body>
    </html>
    """
    
    return send_email([admin_email], subject, message, html_message)

def notify_error(error_message: str, admin_email: str) -> bool:
    """
    Send notification when an error occurs during training.
    
    Args:
        error_message: The error message to include
        admin_email: Email address to send notification to
        
    Returns:
        bool: True if notification sent successfully
    """
    subject = "PINN Training Error"
    message = f"""
    An error occurred during PINN training:
    
    Error: {error_message}
    
    Please check your logs and workspace.
    """
    
    return send_email([admin_email], subject, message) 