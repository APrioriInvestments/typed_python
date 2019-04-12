/******************************************************************************
   Copyright 2017-2019 Nativepython Authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

#pragma once

/**********
Views provide running (native)python threads with a snapshotted view of the current
object-database's subscribed data. A view holds a reference to a collection of objects
that have different representation at different transaction_ids, along with a single
transaction_id. We show a coherent view of all the objects whose versions are <= the
given transaction_id.

We also track all objects and indices we read or write during execution.
***********/

class View {
public:

};