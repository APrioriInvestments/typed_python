#   Copyright 2018 Braxton Mckee
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
#   limitations under the License.

import traceback
import logging
import time
import boto3
import uuid
import base64
import json
import sys
import time
import os.path
import logging
import traceback
import datetime
import os

from object_database import ServiceBase, service_schema, Schema, Indexed
from nativepython.python.string_util import closest_N_in

schema = Schema("core.AwsWorkerBootService")

valid_instance_types = (
    "t2.nano", "t2.micro", "t2.small", "t2.medium", "t2.large", 
    "t2.xlarge", "t2.2xlarge", "t3.nano", "t3.micro", "t3.small", 
    "t3.medium", "t3.large", "t3.xlarge", "t3.2xlarge", "m4.large", 
    "m4.xlarge", "m4.2xlarge", "m4.4xlarge", "m4.10xlarge", "m4.16xlarge", 
    "m5.large", "m5.xlarge", "m5.2xlarge", "m5.4xlarge", "m5.12xlarge", 
    "m5.24xlarge", "m5d.large", "m5d.xlarge", 
    "m5d.2xlarge", "m5d.4xlarge", "m5d.12xlarge", "m5d.24xlarge"
    "c4.large", "c4.xlarge", "c4.2xlarge", "c4.4xlarge", "c4.8xlarge", 
    "c5.large", "c5.xlarge", "c5.2xlarge", "c5.4xlarge", "c5.9xlarge", 
    "c5.18xlarge", "c5d.xlarge", "c5d.2xlarge", "c5d.4xlarge", "c5d.9xlarge", "c5d.18xlarge"
    "r4.large", "r4.xlarge", "r4.2xlarge", "r4.4xlarge", "r4.8xlarge", 
    "r4.16xlarge", "r5.large", "r5.xlarge", "r5.2xlarge", "r5.4xlarge", 
    "r5.12xlarge", "r5.24xlarge", "r5d.large", "r5d.xlarge", "r5d.2xlarge", 
    "r5d.4xlarge", "r5d.12xlarge", "r5d.24xlarge", "x1.16xlarge", "x1.32xlarge", 
    "x1e.xlarge", "x1e.2xlarge", "x1e.4xlarge", "x1e.8xlarge", "x1e.16xlarge", 
    "x1e.32xlarge", "z1d.large", "z1d.xlarge", 
    "z1d.2xlarge", "z1d.3xlarge", "z1d.6xlarge", "z1d.12xlarge"
    "d2.xlarge", "d2.2xlarge", "d2.4xlarge", "d2.8xlarge", "h1.2xlarge", 
    "h1.4xlarge", "h1.8xlarge", "h1.16xlarge", "i3.large", "i3.xlarge", 
    "i3.2xlarge", "i3.4xlarge", "i3.8xlarge", "i3.16xlarge"
    )

@schema.define
class Configuration:
    db_hostname = str          #hostname to connect back to
    db_port = int              #port to connect back to
    region = str               #region to boot into
    vpc_id = str               #id of vpc to boot into
    subnet = str               #id of subnet to boot into
    security_group = str       #id of security group to boot into
    keypair = str              #security keypair name to use
    worker_name = str          #name of workers. This should be unique to this install.
    worker_iam_role_name = str #AIM role to boot workers into
    linux_ami = str            #default linux AMI to use when booting linux workers
    defaultStorageSize =  int  #gb of disk to mount on booted workers (if they need ebs)

    max_to_boot = int          #maximum number of workers we'll boot

@schema.define
class State:
    instance_type = Indexed(str)

    booted = int
    desired = int

own_dir = os.path.split(__file__)[0]

linux_bootstrap_script = open(os.path.join(own_dir, "aws_linux_bootstrap.sh"), "r").read()

class AwsApi:
    def __init__(self):
        self.config = Configuration.lookupOne()

        self.ec2 = boto3.resource('ec2',region_name=self.config.region)
        self.ec2_client = boto3.client('ec2',region_name=self.config.region)
        self.s3 = boto3.resource('s3',region_name=self.config.region)
        self.s3_client = boto3.client('s3',region_name=self.config.region)

    def allRunningInstances(self, includePending=True):
        filters = [{  
            'Name': 'tag:Name',
            'Values': [self.config.worker_name]
            }]

        res = {}

        for reservations in self.ec2_client.describe_instances(Filters=filters)["Reservations"]:
            for instance in reservations["Instances"]:
                if instance['State']['Name'] in ('running','pending') if includePending else ('running',):
                    res[str(instance["InstanceId"])] = instance

        return res

    def isInstanceWeOwn(self, instance):
        #make sure this instance is definitely one we booted.

        if not [t for t in instance.tags if t["Key"] == "Name" and t["Value"] == self.config.worker_name]:
            return False

        if instance.subnet.id != self.config.subnet:
            return False

        if not [t for t in instance.security_groups if t['GroupId'] == self.config.security_group]:
            return False

        if instance.key_pair.name != self.config.keypair:
            return False
        
        return True

    def terminateInstanceById(self, id):
        instance = self.ec2.Instance(id)
        assert self.isInstanceWeOwn(instance)
        logging.info("Terminating AWS instance %s", instance)
        instance.terminate()

    def bootWorker(self, 
            instanceType,
            clientToken=None,
            amiOverride=None,
            bootScriptOverride=None,
            nameValueOverride=None,
            extraTags=None,
            wantsTerminateOnShutdown=True
            ):
        boot_script = (
            linux_bootstrap_script.format(
                db_hostname=self.config.db_hostname,
                db_port=self.config.db_port
                )
            )

        if clientToken is None:
            clientToken = str(uuid.uuid4())

        if amiOverride is not None:
            ami = amiOverride
        else:
            ami = self.config.linux_ami

        def has_ephemeral_storage(instanceType):
            for t in ['m3', 'c3', 'x1', 'r3', 'f1', 'h1', 'i3', 'd2']:
                if instanceType.startswith(t):
                    return True
            return False

        if has_ephemeral_storage(instanceType):
            deviceMapping = {
                'DeviceName': '/dev/xvdb',
                'VirtualName': "ephemeral0"
                }
        else:
            deviceMapping = {
                'DeviceName': '/dev/xvdb',
                'VirtualName': "ephemeral0",
                "Ebs": {
                    "Encrypted": False,
                    "DeleteOnTermination": True,
                    "VolumeSize": self.config.defaultStorageSize,
                    "VolumeType": "gp2"
                    }
                }

        nameValue = nameValueOverride or self.config.worker_name

        return str(self.ec2.create_instances(
            ImageId=ami,
            InstanceType=instanceType,
            KeyName=self.config.keypair,
            MaxCount=1,
            MinCount=1,
            SecurityGroupIds=[self.config.security_group],
            SubnetId=self.config.subnet,
            ClientToken=clientToken,
            InstanceInitiatedShutdownBehavior='terminate' if wantsTerminateOnShutdown else "stop",
            IamInstanceProfile={'Name': self.config.worker_iam_role_name},
            UserData=boot_script, #base64.b64encode(boot_script.encode("ASCII")),
            BlockDeviceMappings=[deviceMapping],
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [{ 
                        "Key": 'Name', 
                        "Value": nameValue
                        }] + [{ "Key": k, "Value": v} for (k,v) in (extraTags or {}).items()]
                }]
            )[0].id)


class AwsWorkerBootService(ServiceBase):
    def __init__(self, db, serviceInstance):
        ServiceBase.__init__(self, db, serviceInstance)

        self.SLEEP_INTERVAL = 10.0

    @staticmethod
    def currentTargets():
        return {s.instance_type: s.desired for s in State.lookupAll()}

    @staticmethod
    def currentBooted():
        return {s.instance_type: s.booted for s in State.lookupAll()}

    @staticmethod
    def setBootState(instance_type, target):
        if instance_type not in valid_instance_types:
            raise Exception(
                "Instance type %s is not a valid instance type. Did you mean one of %s?" % (
                    instance_type,
                    closest_N_in(instance_type, valid_instance_types, 3)
                    )
                )

        s = State.lookupAny(instance_type=instance_type)
        if not s:
            s = State(instance_type=instance_type)
        s.desired = target

    @staticmethod
    def bootOneDirectly(instance_type):
        AwsApi().bootWorker(instance_type)

    @staticmethod
    def shutdownAll():
        for s in State.lookupAll():
            s.desired = 0

    @staticmethod
    def shutOneDown(instance_type):
        api = AwsApi()
        i = [x for x in AwsApi.allRunningInstances().values() if x['InstanceType'] == instance_type]
        if not i:
            raise Exception("No instances of type %s are booted." % instance_type)
        else:
            logging.info("Terminating instance %s", i["InstanceId"])

        AwsApi().terminateInstanceById(i[0])

    @staticmethod
    def configure(
            db_hostname,
            db_port,
            region,
            vpc_id,
            subnet,
            security_group,
            keypair,
            worker_name,
            worker_iam_role_name,
            linux_ami,
            defaultStorageSize,
            max_to_boot
            ):
        c = Configuration.lookupAny()
        if not c:
            c = Configuration()

        if db_hostname is not None:
            c.db_hostname = db_hostname
        if db_port is not None:
            c.db_port = db_port
        if region is not None:
            c.region = region
        if vpc_id is not None:
            c.vpc_id = vpc_id
        if subnet is not None:
            c.subnet = subnet
        if security_group is not None:
            c.security_group = security_group
        if keypair is not None:
            c.keypair = keypair
        if worker_name is not None:
            c.worker_name = worker_name
        if worker_iam_role_name is not None:
            c.worker_iam_role_name = worker_iam_role_name
        if linux_ami is not None:
            c.linux_ami = linux_ami
        if defaultStorageSize is not None:
            c.defaultStorageSize = defaultStorageSize
        if max_to_boot is not None:
            c.max_to_boot = max_to_boot

    def setBootCount(self, instance_type, count):
        state = State.lookupAny(instance_type=instType)
        if not state:
            state = State(instance_type=instType)
        state.desired = count

    def initialize(self):
        with self.db.transaction():
            self.api = AwsApi()

    def doWork(self, shouldStop):
        while not shouldStop.is_set():
            if not self.pushTaskLoopForward():
                time.sleep(1.0)

    def pushTaskLoopForward(self):
        with self.db.view():
            actuallyUsed = self.api.allRunningInstances()

        instanceTypes = {}
        instancesByType = {}

        for machineId, instance in actuallyUsed.items():
            instanceTypes[instance["InstanceType"]] = instanceTypes.get(instance["InstanceType"],0)+1
            instancesByType[instance["InstanceType"]] = instancesByType.get(instance["InstanceType"],())+(instance,)

        for t in instancesByType:
            instancesByType[t] = list(instancesByType[t])


        with self.db.transaction():
            for state in State.lookupAll():
                if state.instance_type not in instanceTypes:
                    state.booted = 0

            for instType, count in instanceTypes.items():
                state = State.lookupAny(instance_type=instType)
                if not state:
                    state = State(instance_type=instType)
                state.booted = count

            for state in State.lookupAll():
                while state.booted > state.desired:
                    logging.info("We have %s instances of type %s booted vs %s desired. Shutting one down.", 
                        state.booted,
                        state.instance_type,
                        state.desired
                        )

                    instance = instancesByType[state.instance_type].pop()
                    self.api.terminateInstanceById(instance["InstanceId"])
                    state.booted -= 1

                while state.booted < state.desired:
                    logging.info("We have %s instances of type %s booted vs %s desired. Booting one.", 
                        state.booted,
                        state.instance_type,
                        state.desired
                        )

                    state.booted += 1

                    self.api.bootWorker(state.instance_type)

        time.sleep(self.SLEEP_INTERVAL)
        

                        
