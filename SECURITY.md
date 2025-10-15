# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of ML Opportunity Prediction System seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please Do Not

- **Do not** open a public GitHub issue for security vulnerabilities
- **Do not** disclose the vulnerability publicly until it has been addressed

### How to Report

**Submit security vulnerabilities via:**
- GitHub Security Advisory (preferred)
- Email: [your-security-email@domain.com]

### What to Include

When reporting a vulnerability, please include:

1. **Description**: Clear description of the vulnerability
2. **Impact**: What could an attacker do?
3. **Affected Components**: Which parts of the system are vulnerable?
4. **Steps to Reproduce**: Detailed steps to reproduce the issue
5. **Proof of Concept**: Code or screenshots if applicable
6. **Suggested Fix**: If you have ideas for fixing it

### Example Report

```markdown
**Vulnerability**: SQL Injection in data query function

**Impact**: Attacker could read/modify database contents

**Affected Component**:
- File: shared/common.py
- Function: execute_query()
- Lines: 45-60

**Steps to Reproduce**:
1. Send malicious input to query parameter
2. Query executes without sanitization
3. Database access gained

**Proof of Concept**:
[Include code or screenshot]

**Suggested Fix**:
Use parameterized queries instead of string concatenation
```

## Response Timeline

- **Initial Response**: Within 48 hours
- **Vulnerability Confirmation**: Within 7 days
- **Fix Development**: Varies by severity
- **Patch Release**: As soon as possible after fix
- **Public Disclosure**: After patch is released

## Security Best Practices

### For Contributors

1. **Never commit secrets**:
   - API keys
   - Passwords
   - Connection strings
   - Certificates
   - Access tokens

2. **Use environment variables** for sensitive data

3. **Validate all inputs** from external sources

4. **Follow secure coding practices**:
   - Parameterized queries
   - Input sanitization
   - Output encoding
   - Least privilege principle

5. **Keep dependencies updated**:
   ```bash
   pip list --outdated
   pip install -U package-name
   ```

### For Deployers

1. **Protect environment variables**:
   - Use Azure Key Vault for secrets
   - Never store secrets in code
   - Rotate credentials regularly

2. **Configure Azure security**:
   - Enable Azure AD authentication
   - Use managed identities
   - Configure network restrictions
   - Enable logging and monitoring

3. **Access control**:
   - Implement principle of least privilege
   - Use role-based access control (RBAC)
   - Audit access logs regularly

4. **Data protection**:
   - Encrypt data at rest
   - Use HTTPS for data in transit
   - Implement data retention policies
   - Anonymize sensitive data

5. **Monitor for vulnerabilities**:
   ```bash
   # Check Python packages for known vulnerabilities
   pip install safety
   safety check

   # Check with Bandit for security issues
   pip install bandit
   bandit -r .
   ```

## Common Security Issues

### 1. Exposed Credentials

**Problem**: API keys or passwords in code

**Solution**:
```python
# ❌ Bad
api_key = "sk-1234567890abcdef"

# ✅ Good
import os
api_key = os.getenv("API_KEY")
```

### 2. SQL Injection

**Problem**: Unsanitized user input in queries

**Solution**:
```python
# ❌ Bad
query = f"SELECT * FROM users WHERE id = {user_id}"

# ✅ Good
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))
```

### 3. Path Traversal

**Problem**: Unrestricted file access

**Solution**:
```python
# ❌ Bad
file_path = f"./data/{user_input}.csv"

# ✅ Good
from pathlib import Path
safe_path = Path("./data").joinpath(user_input).resolve()
if not str(safe_path).startswith(str(Path("./data").resolve())):
    raise ValueError("Invalid path")
```

### 4. Insecure Deserialization

**Problem**: Loading untrusted pickle files

**Solution**:
```python
# ❌ Bad
import pickle
model = pickle.load(open(user_file, 'rb'))

# ✅ Good
# Verify file source and integrity first
# Use safer formats like JSON when possible
import json
config = json.load(open(verified_file, 'r'))
```

## Security Checklist for PRs

Before submitting a pull request, verify:

- [ ] No secrets or credentials in code
- [ ] All user inputs are validated
- [ ] No SQL injection vulnerabilities
- [ ] No path traversal issues
- [ ] Dependencies are up to date
- [ ] Error messages don't leak sensitive info
- [ ] Logging doesn't capture sensitive data
- [ ] HTTPS is used for external calls
- [ ] Authentication is properly implemented
- [ ] Authorization checks are in place

## Dependency Security

### Automated Scanning

We use automated tools to scan for vulnerabilities:

- **Dependabot**: Automated dependency updates
- **Safety**: Python package vulnerability scanner
- **Bandit**: Security issue scanner for Python code

### Manual Review

Regularly review dependencies:

```bash
# List all dependencies
pip freeze > requirements-current.txt

# Check for known vulnerabilities
pip install safety
safety check --file requirements.txt

# Scan code for security issues
pip install bandit
bandit -r . -f html -o bandit-report.html
```

## Azure Security Configuration

### Required Security Settings

1. **Key Vault**:
   - Store all secrets in Azure Key Vault
   - Use managed identities for access
   - Enable soft delete and purge protection

2. **Function App**:
   - Enable HTTPS only
   - Use managed identity
   - Configure CORS appropriately
   - Enable App Service authentication

3. **Storage Account**:
   - Require secure transfer
   - Enable blob encryption
   - Configure firewall rules
   - Use SAS tokens with expiration

4. **Azure ML**:
   - Use compute instance with managed identity
   - Encrypt training data
   - Secure model endpoints with key authentication
   - Enable workspace diagnostics

## Incident Response

If a security incident occurs:

1. **Containment**: Immediately isolate affected systems
2. **Assessment**: Determine scope and impact
3. **Notification**: Inform users if data was compromised
4. **Remediation**: Fix vulnerability and deploy patch
5. **Post-mortem**: Document incident and improve processes

## Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Azure Security Best Practices](https://docs.microsoft.com/en-us/azure/security/)
- [Python Security](https://python.readthedocs.io/en/latest/library/security.html)
- [CWE Top 25](https://cwe.mitre.org/top25/)

## Contact

For security-related questions or concerns:
- GitHub Security Advisories (preferred)
- Email: [your-security-email@domain.com]
- Do not discuss security issues in public forums

---

Thank you for helping keep ML Opportunity Prediction System secure!
