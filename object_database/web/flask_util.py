#   Copyright 2019 Nativepython Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.from collections import defaultdict

from flask import (
    abort,
    has_request_context,
    request,
    url_for,
)

from urllib.parse import urlparse, urljoin, urlencode, parse_qs, urlsplit, urlunsplit


def request_ip_address():
    assert has_request_context()
    return request.remote_addr


def is_safe_url(target, require_https=None, forbid_cross_site=None):

    require_https = False if require_https is None else require_https
    forbid_cross_site = True if forbid_cross_site is None else forbid_cross_site

    allowed_schemes = ('https', ) if require_https else ('https', 'http')

    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))

    cross_site_check = ref_url.netloc == test_url.netloc if forbid_cross_site else True

    return test_url.scheme in allowed_schemes and cross_site_check


def next_url(fallback_url=None, fallback_endpoint='index', next_key='next',
             logger=None, require_https=None, forbid_cross_site=None):
    target_url = request.args.get(next_key)

    if not target_url:
        target_url = fallback_url or url_for(fallback_endpoint)

    if not target_url:
        if logger:
            logger.error("ERROR: failed to resolve target URL during redirect.")
        return abort(400)

    # is_safe_url should check if the url is safe for redirects.
    if not is_safe_url(target_url, require_https=require_https, forbid_cross_site=forbid_cross_site):
        if logger:
            logger.error("ERROR: not safe for redirect: {}".format(target_url))
        return abort(400)

    return target_url


def url_with_request_parameters(url, req=None, filtr=None):
    """ Adds the parameters of the current Flask request to the given url. """
    req = req or request
    for k, v in req.args.items():
        if filtr and k not in filtr:
            continue
        url = set_query_parameter(url, k, v)
    return url


def set_query_parameter(url, param_name, param_value):
    """Given a URL, set or replace a query parameter and return the result. """
    scheme, netloc, path, query_string, fragment = urlsplit(url)
    query_params = parse_qs(query_string)

    query_params[param_name] = [param_value]
    new_query_string = urlencode(query_params, doseq=True)

    return urlunsplit((scheme, netloc, path, new_query_string, fragment))
